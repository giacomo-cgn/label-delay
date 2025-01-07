
from tqdm import tqdm
import datetime
import os

import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
from torchvision.models import resnet18

from ..data import get_benchmark_label_delay, SimpleClassBalancedSampler, SupervisedDataset
from ..buffer import ClassBalancedBuffer
from ..utils import WarmupCosineSchedule
from ..optimizer import init_opt
from ..transforms import make_transforms
from ..evaluate import exec_eval



def train_fgdumb(args, log_folder, device):
    # Transformations
    if 'cifar' in args.dataset or 'tiny' in args.dataset:
        crop_scale = (0.75, 1.0) if args.multicrop > 0 else (0.5, 1.0)
        mc_scale = (0.3, 0.75)
        mc_size = 18
    else:
        crop_scale = (0.14, 1.0) if args.multicrop > 0 else (0.08, 1.0)
        mc_scale = (0.05, 0.14)
        mc_size = 96
    
    
    transforms = make_transforms(
        dataset_name=args.dataset,
        training=True,
        crop_scale=crop_scale,
        basic_augmentations=False,
        color_jitter=args.color_jitter_strength,
        normalize=args.normalize)



    # Get benchmark
    benchmark = get_benchmark_label_delay(dataset_name=args.dataset, dataset_root=args.dataset_root, 
                              eval_transform=transforms,
                              num_exps=args.num_exps, valid_ratio=args.valid_ratio,
                              supervised_ratio=args.supervised_ratio, seed=args.dataset_seed,
                              unsup_anticipate_ratio=args.unsup_anticipate_ratio, drop_last_exp=args.drop_last_exp
                            )
    
    #  Init model
    encoder = resnet18(num_classes=benchmark.num_classes)
    encoder = encoder.to(device)

    # Initialize optimizer
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16)
    encoder, optimizer = init_opt(
        optimizer_name=args.optimizer,
        encoder=encoder,
        weight_decay=args.weight_decay,
        ref_lr=args.lr,
        nesterov=args.nesterov,
        ref_lr_cls=args.lr_cls)


    # Init buffer
    buffer = ClassBalancedBuffer(args.buffer_size)
    num_seen_classes = 0
    seen_classes = []

    for tr_exp_idx, (sup_exp, unsup_exp) in enumerate(zip(benchmark.supervised_tr_stream, benchmark.unsupervised_tr_stream)):
        print(f'###### Starting Task {tr_exp_idx} ######')
        pre_classes = seen_classes
        cur_classes = benchmark.exp_classes_list[tr_exp_idx]
        seen_classes += cur_classes
        num_cur_classes = len(cur_classes)
        num_pre_classes = len(pre_classes)
        num_seen_classes = len(seen_classes)

        if tr_exp_idx > 0:
            # Update buffer 
            buffer.update(sup_exp)
            # For subsequent tasks, we use only class balanced buffer
            sup_dataset = SupervisedDataset(data=buffer, transforms=transforms, num_views=args.supervised_views)
        elif tr_exp_idx == 0:
            # For first task, we use only the first task data
            print(f'###### Using only first task data for first task ######')
            sup_dataset = SupervisedDataset(data=sup_exp, transforms=transforms, num_views=args.supervised_views)
        
        print('LEN SUP DATASET:', len(sup_dataset))
        sup_loader = DataLoader(sup_dataset, num_workers=args.num_workers, batch_size=args.sup_mb_size, shuffle=True)

        ipe = len(sup_loader) # iterations per epoch
        #  Init scheduler
        scheduler = WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=args.warmup*ipe,
            start_lr=args.start_lr,
            ref_lr=args.lr,
            T_max=args.epochs*ipe,
            final_lr=args.final_lr)

        if tr_exp_idx > 0:
            print('Freezing all layers except clasification layer!')
            # Freeze all layers except clasification layer
            for param in encoder.parameters():
                param.requires_grad = False
            for param in encoder.fc.parameters():
                param.requires_grad = True


        # Train
        encoder.train()
        for epoch in range(args.epochs):
            print(f'Doing epoch {epoch}')

            # Iterate over dataloaders
            print('len suploader:', len(sup_loader))
            for sdata in tqdm(sup_loader):               

                slabels = sdata[1].to(device, non_blocking=True).repeat(args.supervised_views)
                # print(f'slabels.shape: {slabels.shape}')
                imgs = [s.to(device, non_blocking=True) for s in sdata[0]]
                imgs = torch.cat(imgs, dim=0)

                with torch.cuda.amp.autocast(enabled=args.use_fp16):
                    optimizer.zero_grad()
    				# Forward pass, l=logits
                    l = encoder(imgs)

                    # Compute loss in full precision
                    with torch.cuda.amp.autocast(enabled=False):

                        # Step 1. convert representations to fp32
                        l = l.float()                                                
                        online_eval_loss = F.cross_entropy(l, slabels)

                    loss = online_eval_loss

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

        # Update buffer after first experience
        if tr_exp_idx == 0:
            print('Updating buffer...')
            buffer.update(sup_exp)

        # Checkpoint at the end of each experience
        chkpt_pth = os.path.join(log_folder, 'checkpoints')
        if not os.path.exists(chkpt_pth):
            os.makedirs(chkpt_pth)
        torch.save({
            'encoder': encoder.state_dict()
            }, os.path.join(chkpt_pth, f'checkpoint_trexp{tr_exp_idx}.pth'))


        # Eval
        exec_eval(encoder=encoder, test_stream=benchmark.test_stream, transforms=transforms, tr_exp_idx=tr_exp_idx,
                  val_stream=benchmark.valid_stream, device=device, log_folder=log_folder, torch_resnet=True)