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
from ..utils.data import class_weights

cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"model_dependencies")
class RSTModel(BaseModel):
    def __init__(self, arch: str = "bert-base-uncased", max_length: int = 60, num_action_classes: int = 3,
                 num_nuclearity_classes: int = 3, num_relation_classes: int = 18, blstm_hidden_size: int = 100, rnn_layers: int = 1,
                 dropout: float = 0.0, glove_dim: int = 300, edu_encoder_arch: str = "bert", use_glove: bool = False, pooler: str = "mean",
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
        self.edu_encoder_arch = edu_encoder_arch
        self.blstm_hidden_size = blstm_hidden_size
        self.use_glove = use_glove
        self.pooler = pooler
        # EDU representation
        # self.tokenizer = AutoTokenizer.from_pretrained(arch)
        if edu_encoder_arch == "bert":
            self.edu_encoder = AutoModel.from_pretrained(arch)
            # self.layernorm = nn.LayerNorm(self.edu_encoder.config.hidden_size)
            self.edu_bi_encoder = nn.LSTM(self.edu_encoder.config.hidden_size, blstm_hidden_size, rnn_layers,
                                          batch_first=True, dropout=(0 if rnn_layers == 1 else dropout),
                                          bidirectional=True)
        elif edu_encoder_arch == "elmo":
            elmo_model_size = embedding.keywords['model_size']
            elmo_dim = embedding.keywords[f'{elmo_model_size}_word_dim']
            if not os.path.exists(os.path.join(cache_dir, 'elmo', elmo_model_size, 'weights.hdf5')):
                logging.info("Downloading Elmo model...")
                os.makedirs(os.path.join(cache_dir, 'elmo', elmo_model_size), exist_ok=True)
                if elmo_model_size == 'small':
                    remote_weight_file = embedding.keywords['small_weight_file']
                    remote_options_file = embedding.keywords['small_options_file']

                elif elmo_model_size == 'medium':
                    remote_weight_file = embedding.keywords['medium_weight_file']
                    remote_options_file = embedding.keywords['medium_options_file']
                elif elmo_model_size == 'large':
                    remote_weight_file = embedding.keywords['large_weight_file']
                    remote_options_file = embedding.keywords['large_options_file']
                else:
                    raise ValueError("Elmo model size not supported!")
                torch.hub.download_url_to_file(remote_weight_file, os.path.join(cache_dir, 'elmo', elmo_model_size, 'weights.hdf5'))
                torch.hub.download_url_to_file(remote_options_file, os.path.join(cache_dir, 'elmo', elmo_model_size, 'options.json'))

            options_file = os.path.join(cache_dir, 'elmo', elmo_model_size, 'options.json')
            weight_file = os.path.join(cache_dir, 'elmo', elmo_model_size, 'weights.hdf5')
            num_output_representations = embedding.keywords['num_output_representations']
            self.num_output_representations = num_output_representations
            dropout = embedding.keywords['dropout']
            requires_grad = embedding.keywords['requires_grad']
            do_layer_norm = embedding.keywords['do_layer_norm']
            self.edu_encoder = Elmo(options_file=options_file, weight_file=weight_file, num_output_representations=num_output_representations,
                             dropout=dropout, requires_grad=requires_grad, do_layer_norm=do_layer_norm)

            if use_glove:
                input_size = elmo_dim + glove_dim
            else:
                input_size = elmo_dim

            self.batchnorm = nn.BatchNorm1d(
                input_size, affine=False, track_running_stats=False
            )
            # span representation
            self.edu_bi_encoder = nn.LSTM(input_size, blstm_hidden_size, rnn_layers,
                                          batch_first=True, dropout=(0 if rnn_layers == 1 else dropout), bidirectional=True)
        else:
            raise NotImplementedError

        # action model
        self.action_left_encoder = nn.Linear(blstm_hidden_size * 2, blstm_hidden_size, bias=False)
        self.action_right_encoder = nn.Linear(blstm_hidden_size * 2, blstm_hidden_size, bias=False)
        self.action_header = nn.Linear(blstm_hidden_size * 2, 1, bias=False)

        # nuclearity model
        self.nuclearity_left_encoder = nn.Linear(blstm_hidden_size * 2, blstm_hidden_size, bias=False)
        self.nuclearity_right_encoder = nn.Linear(blstm_hidden_size * 2, blstm_hidden_size, bias=False)
        # self.nuclearity_left_header = nn.Linear(blstm_hidden_size, num_nuclearity_classes, bias=False)
        # self.nuclearity_right_header = nn.Linear(blstm_hidden_size, num_nuclearity_classes, bias=False)
        self.nuclearity_header = nn.Linear(blstm_hidden_size * 2, num_nuclearity_classes, bias=False)

        # relation model
        self.relation_left_encoder = nn.Linear(blstm_hidden_size * 2, blstm_hidden_size, bias=False)
        self.relation_right_encoder = nn.Linear(blstm_hidden_size * 2, blstm_hidden_size, bias=False)
        # self.relation_left_header = nn.Linear(blstm_hidden_size, num_relation_classes, bias=False)
        # self.relation_right_header = nn.Linear(blstm_hidden_size, num_relation_classes, bias=False)

        self.relation_header = nn.Linear(blstm_hidden_size * 2, num_relation_classes, bias=False)
        # self.weight_bilateral = nn.Bilinear(
        #     blstm_hidden_size, blstm_hidden_size, num_relation_classes, bias=False
        # )
        self.dropout = nn.Dropout(dropout)

        assert num_freeze_layers >= 0
        if num_freeze_layers > 0:
            freeze_layers(self, num_freeze_layers)

        self.model_dict = {
            'edu_bi_encoder': self.edu_bi_encoder,
            'action_left_encoder': self.action_left_encoder,
            'action_right_encoder': self.action_right_encoder,
            'action_header': self.action_header,
            'nuclearity_left_encoder': self.nuclearity_left_encoder,
            'nuclearity_right_encoder': self.nuclearity_right_encoder,
            # 'nuclearity_left_header': self.nuclearity_left_header,
            # 'nuclearity_right_header': self.nuclearity_right_header,
            'nuclearity_header': self.nuclearity_header,
            'relation_left_encoder': self.relation_left_encoder,
            'relation_right_encoder': self.relation_right_encoder,
            # 'relation_left_header': self.relation_left_header,
            # 'relation_right_header': self.relation_right_header,
            # 'weight_bilateral': self.weight_bilateral,
            'relation_header': self.relation_header,
        }
        if pooler == 'partial':
            self.u_first = nn.Linear(blstm_hidden_size * 4, 1, bias=False)
            self.u_last = nn.Linear(blstm_hidden_size * 4, 1, bias=False)
            self.model_dict['u_first'] = self.u_first
            self.model_dict['u_last'] = self.u_last

        self.save_outputs = save_outputs
        self.exp_id = exp_id

        self.measure_attrs_runtime = measure_attrs_runtime

    def bert_edu_forward(self, input_ids, attn_mask, mini_batch_size):
        # split inputs and attention_mask into mini-batch

        if mini_batch_size:
            dataset = {'input_ids': input_ids, 'attention_mask': attn_mask}
            dataset = EDUDataset(dataset, mini_batch_size)
            data_loader = DataLoader(dataset, batch_size=mini_batch_size, collate_fn=dataset.collater)
            enc_batches = []
            for batch in data_loader:
                emb = self.edu_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                if self.pooler == 'partial':
                    enc_batches.append(emb.last_hidden_state)
                else:
                    enc_batches.append(emb.pooler_output)
            enc = torch.cat(enc_batches, dim=0)
        else:
            emb = self.edu_encoder(input_ids=input_ids, attention_mask=attn_mask)
            if self.pooler == 'partial':
                enc = emb.last_hidden_state
            else:
                enc = emb.pooler_output

        output, (h_n, c_n) = self.edu_bi_encoder(enc)
        if self.pooler == 'partial':
            h_first = torch.cat((output[:, [0], :], output[:, [0], :]), dim=2).repeat(1, output.size(1), 1)
            h_last = torch.cat((output[:, [-2], :], output[:, [-1], :]), dim=2).repeat(1, output.size(1), 1)
            sGate = self.u_first(h_first) + self.u_last(h_last)
            outputs = torch.mul(output, sGate)
            outputs, max_indices = torch.max(outputs, dim=1)
            outputs = self.dropout(outputs)
            f_emb = torch.cat((torch.zeros((1, self.edu_bi_encoder.hidden_size)).to(outputs.device),
                               outputs[:, :self.edu_bi_encoder.hidden_size]), dim=0)
            b_emb = torch.cat((outputs[:, self.edu_bi_encoder.hidden_size:],
                               torch.zeros((1, self.edu_bi_encoder.hidden_size)).to(outputs.device)), dim=0)

        elif self.pooler == 'mean':

            f_emb = output[:, :self.edu_bi_encoder.hidden_size]
            b_emb = output[:, self.edu_bi_encoder.hidden_size:]
        else:
            raise NotImplementedError
        return f_emb, b_emb

    def elmo_edu_forward(self, input_ids, attn_mask, mini_batch_size, glove_embs, character_ids):

        # split EDUs into mini-batch
        if mini_batch_size:
            dataset = {'input_ids': input_ids, 'attention_mask': attn_mask, 'glove_embs': glove_embs, 'character_ids': character_ids}
            dataset = EDUDataset(dataset, mini_batch_size)
            data_loader = DataLoader(dataset, batch_size=mini_batch_size, collate_fn=dataset.collater)

            enc_batches = []

            for batch in data_loader:
                elmo_embs = self.edu_encoder(batch['character_ids'])['elmo_representations']
                if self.num_output_representations == 3:
                    elmo_embs = (elmo_embs[0]+elmo_embs[1]+elmo_embs[2])/3
                elif self.num_output_representations == 2:
                    elmo_embs = (elmo_embs[0]+elmo_embs[1])/2
                elif self.num_output_representations == 1:
                    elmo_embs = elmo_embs[0]
                else:
                    raise ValueError("num_output_representations must be 1, 2, or 3")
                if self.use_glove:
                    glove_embs = batch['glove_embs']
                    if self.pooler == 'mean':
                        glove_embs = glove_embs.mean(dim=1)
                        elmo_embs = elmo_embs.mean(dim=1)
                    edu_embs = torch.cat((glove_embs, elmo_embs), dim=-1)
                else:
                    if self.pooler == 'mean':
                        elmo_embs = elmo_embs.mean(dim=1)
                    edu_embs = elmo_embs
                enc_batches.append(edu_embs)
            enc = torch.cat(enc_batches, dim=0)
        else:
            elmo_embs = self.edu_encoder(character_ids)['elmo_representations']
            if self.num_output_representations == 3:
                elmo_embs = (elmo_embs[0] + elmo_embs[1] + elmo_embs[2]) / 3
            elif self.num_output_representations == 2:
                elmo_embs = (elmo_embs[0] + elmo_embs[1]) / 2
            elif self.num_output_representations == 1:
                elmo_embs = elmo_embs[0]
            else:
                raise ValueError("num_output_representations must be 1, 2, or 3")
            if self.use_glove:
                if self.pooler == 'mean':
                    glove_embs = glove_embs.mean(dim=1)
                    elmo_embs = elmo_embs.mean(dim=1)
                edu_embs = torch.cat((glove_embs, elmo_embs), dim=-1)
            else:
                if self.pooler == 'mean':
                    elmo_embs = elmo_embs.mean(dim=1)
                edu_embs = elmo_embs
            enc = edu_embs

        if self.pooler == 'partial':
            enc = enc.permute(0, 2, 1)
            enc = self.batchnorm(enc)
            enc = enc.permute(0, 2, 1)
            enc = self.dropout(enc)
            output, (h_n, c_n) = self.edu_bi_encoder(enc)
            h_first = torch.cat((output[:, [0], :], output[:, [0], :]), dim=2).repeat(1, output.size(1), 1)
            h_last = torch.cat((output[:, [-2], :], output[:, [-1], :]), dim=2).repeat(1, output.size(1), 1)
            sGate = self.u_first(h_first) + self.u_last(h_last)
            outputs = torch.mul(output, sGate)
            outputs, max_indices = torch.max(outputs, dim=1)
            outputs = self.dropout(outputs)
            f_emb = torch.cat((torch.zeros((1, self.edu_bi_encoder.hidden_size)).to(outputs.device),
                               outputs[:, :self.edu_bi_encoder.hidden_size]), dim=0)
            b_emb = torch.cat((outputs[:, self.edu_bi_encoder.hidden_size:],
                               torch.zeros((1, self.edu_bi_encoder.hidden_size)).to(outputs.device)), dim=0)
        else:
            enc = self.dropout(enc)
            output, (h_n, c_n) = self.edu_bi_encoder(enc)
            f_emb = output[:, :self.edu_bi_encoder.hidden_size]
            b_emb = output[:, self.edu_bi_encoder.hidden_size:]
        return f_emb, b_emb

    def action_forward(self, start, end ,cut, f_emb, b_emb):

        left_emb = torch.cat((f_emb[cut, :] - f_emb[start - 1, :], b_emb[start - 1, :] - b_emb[cut, :]),
                           dim=0)
        right_emb = torch.cat((f_emb[end, :] - f_emb[cut, :], b_emb[cut, :] - b_emb[end, :]), dim=0)
        left_emb = self.action_left_encoder(left_emb)
        right_emb = self.action_right_encoder(right_emb)
        total_emb = torch.cat((left_emb, right_emb), dim=0)
        total_emb = self.dropout(total_emb)
        total_emb = F.relu(total_emb)
        output = self.action_header(total_emb)
        return output

    def nuclearity_forward(self, start, end ,cut, f_emb, b_emb):
        left_emb = torch.cat((f_emb[cut, :] - f_emb[start - 1, :], b_emb[start - 1, :] - b_emb[cut, :]),
                           dim=0)
        right_emb = torch.cat((f_emb[end, :] - f_emb[cut, :], b_emb[cut, :] - b_emb[end, :]), dim=0)
        left_emb = self.nuclearity_left_encoder(left_emb)
        # left_emb = F.relu(left_emb)
        right_emb = self.nuclearity_right_encoder(right_emb)
        # right_emb = F.relu(right_emb)
        total_emb = torch.cat((left_emb, right_emb), dim=0)
        total_emb = self.dropout(total_emb)
        # total_emb = F.relu(total_emb)
        # left_emb = total_emb[: self.blstm_hidden_size]
        # right_emb = total_emb[self.blstm_hidden_size:]
        output = self.nuclearity_header(total_emb)
        # output = self.nuclearity_left_header(left_emb) + self.nuclearity_right_header(right_emb)
        return output

    def relation_forward(self, start, end ,cut, f_emb, b_emb):
        left_emb = F.elu(torch.cat((f_emb[cut, :] - f_emb[start - 1, :], b_emb[start - 1, :] - b_emb[cut, :]),
                           dim=0))
        right_emb = F.elu(torch.cat((f_emb[end, :] - f_emb[cut, :], b_emb[cut, :] - b_emb[end, :]), dim=0))

        left_emb = self.relation_left_encoder(left_emb)
        # left_emb = F.relu(left_emb)
        right_emb = self.relation_right_encoder(right_emb)
        # right_emb = F.relu(right_emb)

        total_emb = torch.cat((left_emb, right_emb), 0)
        total_emb = self.dropout(total_emb)
        # left_emb = union[: self.blstm_hidden_size]
        # right_emb = union[self.blstm_hidden_size:]
        # output = (
        #         self.weight_bilateral(left_emb, right_emb)
        #         + self.relation_left_header(left_emb)
        #         + self.relation_right_header(right_emb)
        # )

        # total_emb = torch.cat((left_emb, right_emb), dim=0)
        # total_emb = F.relu(total_emb)
        output = self.relation_header(total_emb)
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
        if self.edu_encoder_arch == 'bert':
            f_embs, b_embs = self.bert_edu_forward(input_ids, attn_mask, mini_batch_size)
        elif self.edu_encoder_arch == 'elmo':
            f_embs, b_embs = self.elmo_edu_forward(input_ids, attn_mask, mini_batch_size, glove_embs, character_ids)
        else:
            raise NotImplementedError


        ret_dict, loss_dict,  = {}, {}
        step_loss_dict = collections.defaultdict(list)
        targets = []
        predictions = []
        min_id = 1
        max_id = input_ids.shape[0]
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
                    action_target = torch.tensor([1], device=actions.device) if action else torch.tensor([0], device=actions.device)
                    action_loss = calc_loss(action_logits, action_target, mode='bce')
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
                        rel_loss = calc_loss(rel_logits, rel_target, class_weights=torch.tensor(class_weights).to(input_ids.device))
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
                        # pred_action = torch.argmax(F.softmax(action_logits, dim=0))
                        #
                        pred_action = F.sigmoid(action_logits) > 0.5
                        # if end == max_id or action_dict[pred_action.item()] == "Reduce":
                        if end == max_id or pred_action:
                            action = 1
                            processing_spans = processing_spans[:-2]
                            processing_spans.append((start, end))
                        else:
                            action = 0
                            processing_spans.append((cur_eid, cur_eid))
                            cur_eid += 1

                        if action:
                            nuc_logits = self.nuclearity_forward(max(0, start-1), end-1, cut-1, f_embs, b_embs)
                            pred_nuclearity = torch.argmax(F.softmax(nuc_logits, dim=0))
                            rel_logits = self.relation_forward(max(0, start-1), end-1, cut-1, f_embs, b_embs)


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
        if self.save_outputs and split != 'pred':
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
