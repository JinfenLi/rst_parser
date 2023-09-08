import collections
import logging
import os
import pickle

import allennlp
import torch
from allennlp.modules import Elmo
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .base_model import BaseModel
from ..data.data import EDUDataset
from ..utils.losses import calc_loss
from ..utils.metrics import init_best_metrics, init_perf_metrics, calc_f1
from ..utils.optim import setup_optimizer_params, setup_scheduler, freeze_layers
from ..utils.logging import log_step_losses, log_epoch_losses, log_epoch_metrics, log_step_metrics
from ..utils.data import action_dict

cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"model_dependencies")
class RSTModel(BaseModel):
    def __init__(self, arch: str = "bert-base-uncased", max_length: int = 60, num_action_classes: int = 3,
                 num_nuclearity_classes: int = 3, num_relation_classes: int = 18, blstm_hidden_size: int = 100, rnn_layers: int = 1,
                 dropout: float = 0.0, elmo_dim: int = 256, glove_dim: int = 300,
                 dataset: str = None, num_freeze_layers: int = 0, freeze_epochs=-1, neg_weight=1,
                 save_outputs: str = None, exp_id: str = None,
                 measure_attrs_runtime: bool = False, optimizer: torch.optim.Optimizer = None,
                 scheduler: torch.optim.lr_scheduler = None, embedding: allennlp.modules.Elmo = None, **kwargs):


        super().__init__()

        self.save_hyperparameters(logger=False)

        self.arch = arch
        self.dataset = dataset
        self.max_length = max_length

        self.freeze_epochs = freeze_epochs
        self.neg_weight = neg_weight

        self.best_metrics = init_best_metrics()
        self.perf_metrics = init_perf_metrics()

        self.register_buffer('empty_tensor', torch.LongTensor([]))

        # EDU representation
        # self.tokenizer = AutoTokenizer.from_pretrained(arch)
        # self.edu_encoder = AutoModel.from_pretrained(arch)

        if not os.path.exists(os.path.join(cache_dir, 'elmo', 'weight.hdf5')):
            logging.info("Downloading Elmo model...")
            os.makedirs(os.path.join(cache_dir, 'elmo'), exist_ok=True)
            torch.hub.download_url_to_file(self.hparams.embedding.keywords['weight_file'], os.path.join(cache_dir, 'elmo', 'weight.hdf5'))
            torch.hub.download_url_to_file(self.hparams.embedding.keywords['options_file'], os.path.join(cache_dir, 'elmo', 'options.json'))
        options_file = os.path.join(cache_dir, 'elmo', 'options.json')
        weight_file = os.path.join(cache_dir, 'elmo', 'weight.hdf5')
        # with open(os.path.join(cache_dir, 'glove.pkl'), 'rb') as f:
        #     self.glove_dict = pickle.load(f)
        num_output_representations = embedding.keywords['num_output_representations']
        dropout = embedding.keywords['dropout']
        requires_grad = embedding.keywords['requires_grad']
        do_layer_norm = embedding.keywords['do_layer_norm']
        self.edu_encoder = Elmo(options_file=options_file, weight_file=weight_file, num_output_representations=num_output_representations,
                         dropout=dropout, requires_grad=requires_grad, do_layer_norm=do_layer_norm)


        # span representation
        # self.edu_bi_encoder = nn.LSTM(self.edu_encoder.config.hidden_size, blstm_hidden_size, rnn_layers,
        #                             batch_first=True,  dropout=(0 if rnn_layers == 1 else dropout), bidirectional=True)

        self.edu_bi_encoder = nn.LSTM(elmo_dim+glove_dim, blstm_hidden_size, rnn_layers,
                                      batch_first=True, dropout=(0 if rnn_layers == 1 else dropout), bidirectional=True)
        self.edu_bi_header = nn.Linear(blstm_hidden_size * 2, blstm_hidden_size)

        # action model
        self.action_left_encoder = nn.Linear(blstm_hidden_size * 2, blstm_hidden_size)
        self.action_right_encoder = nn.Linear(blstm_hidden_size * 2, blstm_hidden_size)
        self.action_header = nn.Linear(blstm_hidden_size * 2, num_action_classes)

        # nuclearity model
        self.nuclearity_left_encoder = nn.Linear(blstm_hidden_size * 2, blstm_hidden_size)
        self.nuclearity_right_encoder = nn.Linear(blstm_hidden_size * 2, blstm_hidden_size)
        self.nuclearity_header = nn.Linear(blstm_hidden_size * 2, num_nuclearity_classes)

        # relation model
        self.relation_left_encoder = nn.Linear(blstm_hidden_size * 2, blstm_hidden_size)
        self.relation_right_encoder = nn.Linear(blstm_hidden_size * 2, blstm_hidden_size)
        self.relation_header = nn.Linear(blstm_hidden_size * 2, num_relation_classes)

        # self.sigmoid = nn.Sigmoid()

        assert num_freeze_layers >= 0
        if num_freeze_layers > 0:
            freeze_layers(self, num_freeze_layers)

        self.model_dict = {
            'edu_bi_encoder': self.edu_bi_encoder,
            'edu_bi_header': self.edu_bi_header,
            'action_left_encoder': self.action_left_encoder,
            'action_right_encoder': self.action_right_encoder,
            'action_header': self.action_header,
            'nuclearity_left_encoder': self.nuclearity_left_encoder,
            'nuclearity_right_encoder': self.nuclearity_right_encoder,
            'nuclearity_header': self.nuclearity_header,
            'relation_left_encoder': self.relation_left_encoder,
            'relation_right_encoder': self.relation_right_encoder,
            'relation_header': self.relation_header,
        }

        self.save_outputs = save_outputs
        self.exp_id = exp_id

        self.measure_attrs_runtime = measure_attrs_runtime



    # Define a custom collate function for dynamic batching
    def collate_fn(self, batch):
        # Sort sequences by length in descending order
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        inputs, attention_masks = zip(*batch)
        return inputs, attention_masks

    # def edu_forward(self, inputs, attention_mask, mini_batch_size):
    #     # split inputs and attention_mask into mini-batch
    #
    #     if mini_batch_size:
    #         # total_mini_batches = len(inputs) // mini_batch_size
    #         dataset = {'input_ids': inputs, 'attention_mask': attention_mask}
    #         dataset = EDUDataset(dataset, mini_batch_size)
    #         data_loader = DataLoader(dataset, batch_size=mini_batch_size, collate_fn=dataset.collater)
    #
    #         # inputs = inputs.split(mini_batch_size)
    #         # attention_mask = attention_mask.split(mini_batch_size)
    #         enc_batches = []
    #         for batch in data_loader:
    #             # Determine the starting and ending indices for the current mini-batch
    #             # start_idx = batch_idx * mini_batch_size
    #             # end_idx = (batch_idx + 1) * mini_batch_size
    #             # print(inputs[start_idx:end_idx].shape)
    #             # print(attention_mask[start_idx:end_idx].shape)
    #             enc_batches.append(self.edu_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).pooler_output)
    #         enc = torch.cat(enc_batches, dim=0)
    #     else:
    #         enc = self.edu_encoder(input_ids=inputs, attention_mask=attention_mask).pooler_output
    #     output, (h_n, c_n) = self.edu_bi_encoder(enc)
    #     f_emb = output[:, :self.edu_bi_encoder.hidden_size]
    #     b_emb = output[:, self.edu_bi_encoder.hidden_size:]
    #     return f_emb, b_emb

    def edu_forward(self, inputs, attention_mask, glove_embs, character_ids, mini_batch_size):

        # split EDUs into mini-batch

        if mini_batch_size:
            # total_mini_batches = len(inputs) // mini_batch_size
            dataset = {'input_ids': inputs, 'attention_mask': attention_mask, 'glove_embs': glove_embs, 'character_ids': character_ids}
            dataset = EDUDataset(dataset, mini_batch_size)
            data_loader = DataLoader(dataset, batch_size=mini_batch_size, collate_fn=dataset.collater)

            enc_batches = []

            for batch in data_loader:
                # Determine the starting and ending indices for the current mini-batch
                elmo_embs = self.edu_encoder(batch['character_ids'])['elmo_representations']
                elmo_embs = (elmo_embs[0]+elmo_embs[1]+elmo_embs[2])/3
                glove_embs = batch['glove_embs'].mean(dim=1)
                elmo_embs = elmo_embs.mean(dim=1)
                edu_embs = torch.cat((glove_embs, elmo_embs), dim=1)
                enc_batches.append(edu_embs)
            enc = torch.cat(enc_batches, dim=0)
        else:
            elmo_embs = self.edu_encoder(character_ids)['elmo_representations']
            elmo_embs = (elmo_embs[0]+elmo_embs[1]+elmo_embs[2])/3
            glove_embs = glove_embs.mean(dim=1)
            elmo_embs = elmo_embs.mean(dim=1)
            edu_embs = torch.cat((glove_embs, elmo_embs), dim=1)
            enc = edu_embs
        output, (h_n, c_n) = self.edu_bi_encoder(enc)
        f_emb = output[:, :self.edu_bi_encoder.hidden_size]
        b_emb = output[:, self.edu_bi_encoder.hidden_size:]
        return f_emb, b_emb

    def action_forward(self, start, end ,cut, f_emb, b_emb):

        left_emb = torch.cat((f_emb[cut, :] - f_emb[start - 1, :], b_emb[start - 1, :] - b_emb[cut, :]),
                           dim=0)
        right_emb = torch.cat((f_emb[end, :] - f_emb[cut, :], b_emb[cut, :] - b_emb[end, :]), dim=0)
        left_emb = self.action_left_encoder(left_emb)
        left_emb = F.relu(left_emb)
        right_emb = self.action_right_encoder(right_emb)
        right_emb = F.relu(right_emb)
        output = self.action_header(torch.cat((left_emb, right_emb), dim=0))
        return output

    def nuclearity_forward(self, start, end ,cut, f_emb, b_emb):
        left_emb = torch.cat((f_emb[cut, :] - f_emb[start - 1, :], b_emb[start - 1, :] - b_emb[cut, :]),
                           dim=0)
        right_emb = torch.cat((f_emb[end, :] - f_emb[cut, :], b_emb[cut, :] - b_emb[end, :]), dim=0)
        left_emb = self.nuclearity_left_encoder(left_emb)
        left_emb = F.relu(left_emb)
        right_emb = self.nuclearity_right_encoder(right_emb)
        right_emb = F.relu(right_emb)
        output = self.nuclearity_header(torch.cat((left_emb, right_emb), dim=0))
        return output

    def relation_forward(self, start, end ,cut, f_emb, b_emb):
        left_emb = torch.cat((f_emb[cut, :] - f_emb[start - 1, :], b_emb[start - 1, :] - b_emb[cut, :]),
                           dim=0)
        right_emb = torch.cat((f_emb[end, :] - f_emb[cut, :], b_emb[cut, :] - b_emb[end, :]), dim=0)
        left_emb = self.relation_left_encoder(left_emb)
        left_emb = F.relu(left_emb)
        right_emb = self.relation_right_encoder(right_emb)
        right_emb = F.relu(right_emb)
        output = self.relation_header(torch.cat((left_emb, right_emb), dim=0))
        return output

    def run_step(self, batch, split, batch_idx):

        input_ids = batch['input_ids'].squeeze(0)
        attn_mask = batch['attention_mask'].squeeze(0)
        character_ids = batch['character_ids'].squeeze(0)
        glove_embs = batch['glove_embs'].squeeze(0)
        spans = batch['spans'].squeeze(0) if batch['spans'] is not None else None
        actions = batch['actions'].squeeze(0) if batch['actions'] is not None else None
        forms = batch['forms'].squeeze(0) if batch['forms'] is not None else None
        relations = batch['relations'].squeeze(0) if batch['relations'] is not None else None
        eval_split: str = batch['split']
        mini_batch_size = batch['mini_batch_size']
        if split == 'train':
            assert split == eval_split

        f_embs, b_embs = self.edu_forward(input_ids, attn_mask, glove_embs, character_ids, mini_batch_size)


        ret_dict, loss_dict,  = {}, {}
        step_loss_dict = collections.defaultdict(list)
        targets = []
        predictions = []
        min_id = 1
        max_id = f_embs.shape[0]
        processing_spans = [(min_id, min_id)]
        # current edu id
        cur_eid = 2
        if split == 'train':

            while not processing_spans == [(min_id, max_id)]:
                # reduce now
                if len(processing_spans) > 1:
                    start = processing_spans[-2][0]
                    cut = processing_spans[-2][1]
                    end = processing_spans[-1][1]
                    # current span id
                    cur_span = torch.tensor([start, end], device=spans.device)
                    cur_sid = torch.where((spans == cur_span).all(axis=1))
                    if cur_sid[0].shape[0] > 0:
                        action = 1
                        processing_spans = processing_spans[:-2]
                        processing_spans.append((start, end))
                    else:
                        action = 0
                        processing_spans.append((cur_eid, cur_eid))
                        cur_eid += 1

                    action_logits = self.action_forward(max(0, start-1), end-1, cut-1, f_embs, b_embs)
                    # pred_action = action_logits.argmax(dim=1)
                    action_target = torch.tensor([1], device=actions.device) if action else torch.tensor([0], device=actions.device)
                    action_loss = calc_loss(action_logits, action_target)
                    step_loss_dict['action'].append(action_loss)

                    if action:

                        nuc_logits = self.nuclearity_forward(max(0, start-1), end-1, cut-1, f_embs, b_embs)
                        nuc_pred = torch.argmax(F.softmax(nuc_logits, dim=0))
                        nuc_target = forms[cur_sid]
                        nuc_loss = calc_loss(nuc_logits, nuc_target)
                        step_loss_dict['nuclearity'].append(nuc_loss)

                        rel_logits = self.relation_forward(max(0, start-1), end-1, cut-1, f_embs, b_embs)
                        rel_pred = torch.argmax(F.softmax(rel_logits, dim=0))
                        rel_target = relations[cur_sid]
                        rel_loss = calc_loss(rel_logits, rel_target)
                        step_loss_dict['relation'].append(rel_loss)

                        predictions.append((start, end, nuc_pred, rel_pred))
                        targets.append((start, end, nuc_target, rel_target))

                else:
                    processing_spans.append((cur_eid, cur_eid))
                    cur_eid += 1
            loss_dict['loss'] = torch.stack(step_loss_dict['action']).mean() +\
                                torch.stack(step_loss_dict['nuclearity']).mean() +\
                                torch.stack(step_loss_dict['relation']).mean()
            loss_dict['action_loss'] = torch.stack(step_loss_dict['action']).mean()
            loss_dict['nuclearity_loss'] = torch.stack(step_loss_dict['nuclearity']).mean()
            loss_dict['relation_loss'] = torch.stack(step_loss_dict['relation']).mean()
            # Log step losses
            ret_dict = log_step_losses(self, loss_dict, ret_dict, eval_split)


        else:
            with torch.no_grad():
                while not processing_spans == [(min_id, max_id)]:
                    # reduce now
                    if len(processing_spans) > 1:
                        start = processing_spans[-2][0]
                        cut = processing_spans[-2][1]
                        end = processing_spans[-1][1]
                        action_logits = self.action_forward(max(0, start-1), end-1, cut-1, f_embs, b_embs)
                        pred_action = torch.argmax(F.softmax(action_logits, dim=0))

                        if end == max_id or action_dict[pred_action.item()] == "Reduce":
                            action = 1
                            processing_spans = processing_spans[:-2]
                            processing_spans.append((start, end))
                        else:
                            action = 0
                            processing_spans.append((cur_eid, cur_eid))
                            cur_eid += 1

                        if action:
                            nuc_logits = self.nuclearity_forward(max(0, start-1), end-1, cut-1, f_embs, b_embs)
                            rel_logits = self.relation_forward(max(0, start-1), end-1, cut-1, f_embs, b_embs)

                            pred_nuclearity = torch.argmax(F.softmax(nuc_logits, dim=0))
                            pred_relation = torch.argmax(F.softmax(rel_logits, dim=0))
                            predictions.append((start, end, pred_nuclearity, pred_relation))

                    else:
                        processing_spans.append((cur_eid, cur_eid))
                        cur_eid += 1

                if spans is not None:
                    # construct targets based on reduced_sid, spans, forms, relations
                    for sid, (start, end) in enumerate(spans):
                        if forms[sid] != -1:
                            targets.append((start, end, forms[sid], relations[sid]))

        ret_dict['targets'] = torch.tensor(targets).detach() if spans is not None else None
        ret_dict['predictions'] = torch.tensor(predictions).detach()
        ret_dict['eval_split'] = eval_split
        ret_dict['input_ids'] = input_ids.detach()
        if spans is not None:
            metric_dict = calc_f1(ret_dict['predictions'], ret_dict['targets'])
            log_step_metrics(self, metric_dict, split)  # Log step metrics
        return ret_dict

    def aggregate_epoch(self, outputs, split):
        if outputs['targets'] is not None:
            if split == 'train':
                log_epoch_losses(self, outputs, outputs['eval_split'][0])  # Log epoch losses
            log_epoch_metrics(self, outputs, outputs['eval_split'][0])  # Log epoch metrics
        if self.save_outputs:
            for i, predictions in enumerate(outputs['predictions']):
                with open(os.path.join(self.save_outputs, f"{split}_{i}_predictions.txt"), 'w') as file:
                    for prediction in predictions:
                        file.write(f"{prediction[0]} {prediction[1]} {prediction[2]} {prediction[3]}\n")
            import pickle
            with open(os.path.join(self.save_outputs, f"{split}_outputs.pkl"), 'wb') as file:
                pickle.dump(outputs, file)
        results = {}
        if outputs['eval_split'][0] == 'pred':
            input_ids = outputs['input_ids']
            predictions = outputs['predictions']
            results['predictions'] = predictions
            results['input_ids'] = input_ids
        return results



    def configure_optimizers(self):

        optimizer_params = setup_optimizer_params(self.model_dict, self.hparams.optimizer)
        self.hparams.optimizer.keywords['lr'] = self.hparams.optimizer.keywords['lr'] * self.trainer.world_size
        optimizer = self.hparams.optimizer(params=optimizer_params)
        if self.hparams.scheduler['lr_scheduler'] == 'linear_with_warmup':
            scheduler = setup_scheduler(self.hparams.scheduler, self.total_steps, optimizer)
            return [optimizer], [scheduler]
        elif self.hparams.scheduler['lr_scheduler'] == 'fixed':
            return [optimizer]
        else:
            raise NotImplementedError
