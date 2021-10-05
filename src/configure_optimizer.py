import torch 
import madgrad 
from custom_optimizer import *
from custom_scheduler import *
from config import *
from sam import SAM

def fetch_optimizer(model, optimizer = params['OPTIMIZER'], scheduler = params['SCHEDULER']):

    if optimizer == 'ranger':
        optimizer = Ranger(model.parameters(), lr = params['LR'])
    elif optimizer == 'madgrad':
        optimizer = madgrad.MADGRAD(model.parameters(), lr = params['LR'])
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = params['LR'])
    elif optimizer == 'sam':
        optimizer = SAM(model.parameters(), torch.optim.SGD, lr = params['LR'])
    
    if scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = params['T_MAX'],
            eta_min=params['ETA_MIN']
        )
    elif scheduler == 'custom':
        scheduler = CustomScheduler(
            optimizer,
            **params['SCHEDULER_PARAMS']
        )

    return optimizer, scheduler
