from transformers import get_scheduler

no_decay = ['bias', 'LayerNorm.weight']


def setup_optimizer_params(model_dict, optimizer):
    optimizer_parameters = []
    for attr in model_dict.keys():
        param_dict_1 = {
            'params': [p for n, p in model_dict[attr].named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': optimizer.keywords['weight_decay'],
        }
        param_dict_2 = {
            'params': [p for n, p in model_dict[attr].named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        }
        optimizer_parameters.append(param_dict_1)
        optimizer_parameters.append(param_dict_2)

    return optimizer_parameters

def setup_scheduler(scheduler, total_steps, optimizer):
    if scheduler['warmup_updates'] > 1.0:
        warmup_steps = int(scheduler['warmup_updates'])
    else:
        warmup_steps = int(total_steps *
                            scheduler['warmup_updates'])
    print(
        f'\nTotal steps: {total_steps} with warmup steps: {warmup_steps}\n')

    scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scheduler = {
        'scheduler': scheduler,
        'interval': 'step',
        'frequency': 1
    }
    return scheduler

def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_net(module):
    for p in module.parameters():
        p.requires_grad = True

def freeze_layers(model, num_freeze_layers):
    if model.arch == 'bert-base-uncased':
        assert model.edu_encoder is not None

        # Freeze task encoder's embedding layer
        for p in model.edu_encoder.embeddings.parameters():
            p.requires_grad = False

        # Freeze task encoder's encoder layers
        for i in range(num_freeze_layers):
            for p in model.edu_encoder.encoder.layer[i].parameters():
                p.requires_grad = False

    else:
        raise NotImplementedError