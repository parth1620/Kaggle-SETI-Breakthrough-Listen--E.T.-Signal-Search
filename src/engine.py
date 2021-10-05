import torch 
from torch import nn 
from config import *
from timm.utils.agc import adaptive_clip_grad

from tqdm import tqdm 
import numpy as np
from sklearn.metrics import roc_auc_score

import neptune

NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMGIxNDQyZC00MmI4LTQyZmUtYjg5NS01ODUyNzRkZWNlMzEifQ=="
neptune.init('parthdhameliya/seti-comp', api_token=NEPTUNE_API_TOKEN)
neptune.create_experiment('experiment-1-'+params['MODEL_NAME'], params = params)


def train_fn(model, data_loader, optimizer, scheduler, i, swa_model = None):

    model.train()
    if params['FP16']:
        scaler = torch.cuda.amp.GradScaler()
    fin_loss = 0.0
    targets = []
    outputs = []

    tk = tqdm(data_loader, desc = "Epoch" + " [TRAIN] " + str(i+1))

    for t,data in enumerate(tk):
        for k,v in data.items():
            data[k] = v.to(params['DEVICE'])

        if params['SAM'] == True:
            logits1, loss = model(**data)
            loss.backward()
            optimizer.first_step(zero_grad = True)

            adaptive_clip_grad(model.parameters(), clip_factor=0.01, eps=1e-3, norm_type=2.0)

            logits2, loss2 = model(**data)
            loss2.backward()
            optimizer.second_step(zero_grad = True)

            adaptive_clip_grad(model.parameters(), clip_factor=0.01, eps=1e-3, norm_type=2.0)

            targets += data['labels'] 
            outputs += logits1.sigmoid()

        else:
            if params['FP16']:
                with torch.cuda.amp.autocast():
                    optimizer.zero_grad()
                    logits, loss = model(**data)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
            else:
                optimizer.zero_grad()
                logits, loss = model(**data)
                loss.backward()
                optimizer.step()

            targets += data['labels'] 
            outputs += logits.sigmoid()

        fin_loss += loss.item()
        live_loss = fin_loss/(t+1)

        neptune.log_metric('Train-Progress-Live-Loss', live_loss)
        tk.set_postfix({'loss' : '%.6f' %float(live_loss), 'LR' : optimizer.param_groups[0]['lr']})
    
    neptune.log_metric('Lr-Monitor', optimizer.param_groups[0]['lr'])

    if params['SWA'] == True:
        if i+1 >= params['SWA_START']:
            swa_model.update_parameters(model)
        else:
            if scheduler != None:
                scheduler.step()
    else:
        if scheduler != None:
            scheduler.step()

    tr_loss = fin_loss / len(data_loader)

    outputs = torch.cat(outputs)
    targets = torch.cat(targets)
    auc_score = roc_auc_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())

    neptune.log_metric('Train-loss', tr_loss)
    neptune.log_metric('Train-auc', auc_score)
    
    return tr_loss, auc_score

def eval_fn(model, data_loader, i):

    model.eval()
    fin_loss = 0.0
    targets = []
    outputs = []
    tk = tqdm(data_loader, desc = "Epoch" + " [VALID] " + str(i+1))

    with torch.no_grad():
        for t,data in enumerate(tk):
            for k,v in data.items():
                data[k] = v.to(params['DEVICE'])
            logits, loss = model(**data)
            fin_loss += loss.item() 
            live_loss = fin_loss/(t+1)
            neptune.log_metric('Valid-Progress-Live-Loss', live_loss)
            tk.set_postfix({'loss' : '%.6f' %float(live_loss)})

            targets += data['labels'] 
            outputs += logits.sigmoid()

        vl_loss = fin_loss / len(data_loader)

        outputs = torch.cat(outputs)
        targets = torch.cat(targets)

        auc_score = roc_auc_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())

        neptune.log_metric('Valid-loss', vl_loss)
        neptune.log_metric('Valid-auc', auc_score)

        return vl_loss, auc_score