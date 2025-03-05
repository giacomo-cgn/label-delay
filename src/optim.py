import torch
from torch import optim


def get_optim(model, args):

    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in ['bias', 'bn'])], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in ['bias', 'bn'])], 'weight_decay': 0.0}
            # TODO: separate lr for clf (?)
        
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)
    
    return optimizer