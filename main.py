
import datetime
import os

import torch
from src.train import train
from src.parse_args import parse_args



def main(args):
    mask = False

    # Create log folder
    str_now = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
    log_folder_name = f'{args.dataset}_{args.model}_{args.log_name}_{str_now}'
    log_folder = os.path.join(args.log_root, log_folder_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # Save all args
    with open(os.path.join(log_folder, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')

    # Device
    if torch.cuda.is_available():       
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        if args.gpu_idx < torch.cuda.device_count():
            device = torch.device(f"cuda:{args.gpu_idx}")
            print(f"Using GPU {args.gpu_idx}.")
        else:
            device = torch.device("cuda")
            print(f"GPU {args.gpu_idx} is not available, using GPU 0 instead.")
    
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    train(args, log_folder, device)

    
        


if __name__ == '__main__':
    args = parse_args()
    main(args)
