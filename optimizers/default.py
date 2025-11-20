import torch
from torch.optim import AdamW
from optimizers.modified_muon import get_muon_optimizer

class MinLRExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, min_lr=1e-5):
        self.min_lr = min_lr
        super().__init__(optimizer, gamma)

    def get_lr(self):
        lrs = super().get_lr()
        return [max(lr, self.min_lr) for lr in lrs]

def build_optimizer(model, params_dict, lr, type='AdamW'):
    model_parameters = model.parameters()
    named_model_parameters = model.named_parameters()
    parameters_names = []
    parameters_names.append(
        [
            name_param_pair[0]
            for name_param_pair in model.named_parameters()
        ]
    )
    if type == 'AdamW':
        optim = AdamW(
            model_parameters,
            lr=lr,
            betas=params_dict['AdamW_betas'],
            eps=params_dict['AdamW_eps'],
            weight_decay=params_dict['AdamW_weight_decay'],
        )
    elif type == 'MuonAdamW':
        optim = get_muon_optimizer(
            named_model_parameters,
            adamw_betas=params_dict['AdamW_betas'],
            adamw_eps=params_dict['AdamW_eps'],
            lr=lr,
            muon_wd=params_dict['Muon_weight_decay'],
            adamw_wd=params_dict['AdamW_weight_decay'],
            muon_exclude_keys=params_dict["muon_exclude_keys"],
        )
    else:
        raise ValueError('Unknown optimizer type: %s' % type)

    scheduler = MinLRExponentialLR(optim, gamma=params_dict['scheduler']['gamma'], min_lr=params_dict['scheduler']['min_lr'])

    return optim, scheduler