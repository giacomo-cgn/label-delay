from tqdm import tqdm
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from .data import SupervisedDataset

@torch.no_grad()
def evaluate(model, eval_loader, device, torch_resnet=False):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        preds, labels = [], []

        for x, y in tqdm(eval_loader, desc='Eval', leave=False):
            x = x[0].to(device)
            y = y.to(device)

            logits = model.forward_eval(x)
            preds.append(logits.cpu())
            labels.append(y.cpu())

        preds = torch.cat(preds)
        labels = torch.cat(labels)

    acc = (preds.argmax(-1) == labels).float().mean().item()

    return acc*100


class Evaluator():
    def __init__(self, test_stream, transforms, val_stream=None, device='cpu', log_folder='./log', torch_resnet=False):
        self.device = device
        self.torch_resnet = torch_resnet
        self.log_folder = log_folder

        # Init log file
        with open(os.path.join(log_folder, 'acc.csv'), 'w') as f:
            if val_stream is not None:
                f.write('tr_exp_idx,epoch,test_acc,val_acc\n')
            else:
                f.write('tr_exp_idx,epoch,test_acc\n')

        # Init eval datasets
        self.test_datset = ConcatDataset([SupervisedDataset(test_exp, transforms=transforms, num_views=1) for test_exp in test_stream])
        self.test_loader = DataLoader(self.test_datset, batch_size=256, num_workers=8, pin_memory=True)

        if val_stream is not None:
            self.val_datset = ConcatDataset([SupervisedDataset(val_exp, transforms=transforms, num_views=1) for val_exp in val_stream])
            self.val_loader = DataLoader(self.val_datset, batch_size=256, num_workers=8, pin_memory=True)
        else:
            self.val_loader = None
        
        

    def exec_eval(self, model, tr_exp_idx, epoch):
        test_acc = evaluate(model, self.test_loader, self.device, torch_resnet=self.torch_resnet)
        if self.val_loader is not None:
            val_acc = evaluate(model, self.val_loader, self.device, torch_resnet=self.torch_resnet)
            print(f'Eval at tr_exp {tr_exp_idx}, epoch {epoch}: val_acc {val_acc:.4f}, test_acc {test_acc:.4f}')
            with open(os.path.join(self.log_folder, 'acc.csv'), 'a') as f:
                f.write(f'{tr_exp_idx},{epoch},{test_acc},{val_acc}\n')
        else:
            val_acc = None
            print(f'Eval at tr_exp {tr_exp_idx}, epoch {epoch}: test_acc {test_acc:.4f}')
            with open(os.path.join(self.log_folder, 'acc.csv'), 'a') as f:
                f.write(f'{tr_exp_idx},{epoch},{test_acc}\n')