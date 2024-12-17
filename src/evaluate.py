from tqdm import tqdm
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import SupervisedDataset

@torch.no_grad()
def evaluate(encoder, eval_loader, device):
    encoder.eval()
    preds, labels = [], []

    for x, y in tqdm(eval_loader, desc='Eval', leave=False):
        x = x[0].to(device)
        y = y.to(device)

        _, logits = encoder._forward_backbone(x)

        preds.append(logits.cpu())
        labels.append(y.cpu())

    preds = torch.cat(preds)
    labels = torch.cat(labels)

    acc = (preds.argmax(-1) == labels).float().mean().item()

    return acc*100

def exec_eval(encoder, test_stream, transforms, tr_exp_idx, val_stream=None, device='cpu', log_folder='./log'):
    test_accs = []
    for test_exp_idx, test_exp in enumerate(test_stream):
        test_dataset = SupervisedDataset(test_exp, transforms=transforms, num_views=1)
        test_loader = DataLoader(test_dataset, batch_size=256, num_workers=8, pin_memory=True)

        test_acc = evaluate(encoder, test_loader, device)
        test_accs.append(test_acc)
    avg_test_acc = sum(test_accs)/len(test_accs)
    print(f'Avg Test acc at tr_exp {tr_exp_idx}: {avg_test_acc}')

    val_accs = []
    if val_stream is not None:
        for val_exp_idx, val_exp in enumerate(val_stream):
            val_dataset = SupervisedDataset(val_exp, transforms=transforms, num_views=1)
            val_loader = DataLoader(val_dataset, batch_size=256, num_workers=8, pin_memory=True)

            val_acc = evaluate(encoder, val_loader, device)
            val_accs.append(val_acc)
        avg_val_acc = sum(val_accs)/len(val_accs)
        print(f'Avg Val acc at tr_exp {tr_exp_idx}: {avg_val_acc}')

    save_pth = os.path.join(log_folder, 'results')
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    with open(os.path.join(save_pth, f'accs_trexp{tr_exp_idx}.csv'), 'a') as f:
        if len(test_accs) == len(val_accs):
            f.write('test_exp,val_acc,test_acc\n')
            for test_exp_idx, (val_acc, test_acc) in enumerate(zip(val_accs, test_accs)):
                f.write(f'{test_exp_idx},{val_acc},{test_acc}\n')
            f.write(f'AVG,{avg_val_acc},{avg_test_acc}\n')
        else:
            f.write('test_exp,test_acc\n')
            for test_exp_idx, test_acc in enumerate(test_accs):
                f.write(f'{test_exp_idx},{test_acc}\n')
            f.write(f'AVG,{avg_test_acc}\n')
        