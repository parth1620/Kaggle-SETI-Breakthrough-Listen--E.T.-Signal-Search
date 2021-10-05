import pandas as pd 
import numpy as np 

import torch 
from torch import nn 
from torch.optim.swa_utils import AveragedModel

from data_module import DataModule
from configure_optimizer import *
from models import *
from custom_activation import *
import engine
from sam import SAM

import random 
import os

@torch.no_grad()
def update_bn(loader, model, device=None):
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in tqdm(loader):
        for k,v in input.items():
            input[k] = v.to(device)
        
        model(**input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


def seed_everything(seed=params['SEED']):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def run(folds = params['FOLD']):

    seed_everything()
    df = pd.read_csv(params['TRAIN_CSV'])
    trainloader, validloader = DataModule(df, folds).get_dataloaders()

    model = CNNBiGRU()
    model = model.to(params['DEVICE'])
    #model = replace_activations(model, torch.nn.SiLU, Mish())
    #model = convert_layers(model, nn.BatchNorm2d, nn.GroupNorm, convert_weights=True, num_groups=16)
    #model.load_state_dict(torch.load('/content/drive/MyDrive/SETI_MODELS/final-eca_nfnet_l0-fold-4.pt')['model'])
    print(model)

    if params['SWA'] == True:
        swa_model = AveragedModel(model)
    else:
        swa_model = None

    optimizer, scheduler = fetch_optimizer(model, optimizer = params['OPTIMIZER'], scheduler = params['SCHEDULER'])

    best_score = 0

    for i in range(params['EPOCHS']):

        if params['SWA'] != True:
            tr_loss,tr_auc = engine.train_fn(model, trainloader, optimizer, scheduler, i, swa_model = swa_model)
            vl_loss,vl_auc = engine.eval_fn(model, validloader, i)

            print('tr_auc : {}'.format(tr_auc))
            print('vl_auc : {}'.format(vl_auc))

        else:
            tr_loss,tr_auc = engine.train_fn(model, trainloader, optimizer, scheduler, i, swa_model = swa_model)
            vl_loss,vl_auc = engine.eval_fn(model, validloader, i)

            print('tr_auc : {}'.format(tr_auc))
            print('vl_auc : {}'.format(vl_auc))

        if tr_auc > best_score:
            save_utils = {'model' : model.state_dict(), 'best_score' : best_score, 'epochs' : i+1}
            torch.save(save_utils,  params['MODEL_PATH'] +'NEW-final-'+params['MODEL_NAME']+'-fold-'+str(params['FOLD'])+'.pt')
            best_score = tr_auc


    if params['SWA'] == True:
        torch.save(swa_model.state_dict(), params['MODEL_PATH'] + 'SWA-final-'+params['MODEL_NAME']+'-fold-'+str(folds)+'.pt')
        print("updating bn...")
        update_bn(trainloader, swa_model, device = 'cuda')
        print("Done")

        torch.save(swa_model.state_dict(), params['MODEL_PATH'] + 'SWA-final-'+params['MODEL_NAME']+'-fold-'+str(folds)+'.pt')

run()