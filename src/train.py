
from tqdm import tqdm
import os

import torch
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
import torch.nn.functional as F

from .transforms import make_transforms
from .data import get_benchmark_label_delay, SupervisedDataset, UnsupervisedDataset
from .encoders import get_encoder
from .buffers import ClassBalancedBuffer, ReservoirBuffer
from .scheduler import get_cosine_schedule_with_warmup
from .optim import get_optim
from .evaluate import Evaluator

from .models.hybrid_ssl_semi import HybridSSLSemi
from .models.fixmatch import FixMatch
from .models.fixmatch_unsup_buffer import FixMatchUnsupBuffer



def train(args, log_folder, device):   
    
    labeled_transforms, unlabeled_transforms, eval_transforms = make_transforms(args.dataset, args.use_barlow_transforms)



    # Get benchmark
    benchmark = get_benchmark_label_delay(dataset_name=args.dataset, dataset_root=args.dataset_root, 
                              num_exps=args.num_exps, valid_ratio=args.valid_ratio,
                              supervised_ratio=args.supervised_ratio, seed=args.dataset_seed,
                              delay=args.delay
                            )
    
    #  Init model
    encoder = get_encoder(args.encoder, args.dataset)

    if args.model == 'hybrid_ssl_semi':
        model = HybridSSLSemi(num_classes=benchmark.num_classes,
                            encoder=encoder,
                            proj_hidden_dim=args.proj_hidden_dim,
                            proj_output_dim=args.proj_output_dim,
                            adapter_hidden_dim=args.adapter_hidden_dim,
                            lambda_barlow=args.lambda_barlow,
                            pseudo_label_temp=args.pseudo_label_temp,
                            pseudo_label_thresh=args.pseudo_label_thresh,
                            omega_pseudo=args.omega_pseudo,
                            omega_ssl=args.omega_ssl)
        
    elif args.model == 'fixmatch':
        model = FixMatch(num_classes=benchmark.num_classes,
                         encoder=encoder,
                         pseudo_label_temp=args.pseudo_label_temp,
                         pseudo_label_thresh=args.pseudo_label_thresh,
                         omega_pseudo=args.omega_pseudo)
        
    elif args.model == 'fixmatch_unsup_buffer':
        model = FixMatchUnsupBuffer(num_classes=benchmark.num_classes,
                         encoder=encoder,
                         pseudo_label_temp=args.pseudo_label_temp,
                         pseudo_label_thresh=args.pseudo_label_thresh,
                         omega_pseudo=args.omega_pseudo)
    else:
        raise NotImplementedError(f'Model {args.model} not implemented')
        

    model = model.to(device)

    optimizer = get_optim(model, args)

    # Init log loss file
    with open(os.path.join(log_folder, 'loss.csv'), 'a') as log_loss_file:
        log_loss_file.write('exp_idx,epoch,sup_loss,pseudo_loss,ssl_loss,total_loss\n')

   


    # Init buffers
    buffer_sup = ClassBalancedBuffer(args.s_buffer_size)
    buffer_unsup = ReservoirBuffer(args.u_buffer_size)

    # num_seen_classes = 0
    # seen_classes = []

    # iter_per_task = args.epochs *len(benchmark.unsupervised_tr_stream[0]) // args.unsup_mb_size # assuming tasks of same length

    # Set up evaluator
    evaluator = Evaluator(test_stream=benchmark.test_stream,
                          transforms=eval_transforms,
                           val_stream=benchmark.valid_stream,
                           device=device,
                           log_folder=log_folder,
                           torch_resnet=True)

    for tr_exp_idx, (sup_exp, unsup_exp) in enumerate(zip(benchmark.supervised_tr_stream, benchmark.unsupervised_tr_stream)):
        print(f'###### Starting Task {tr_exp_idx} ######')
        # pre_classes = seen_classes
        # cur_classes = benchmark.exp_classes_list[tr_exp_idx]
        # seen_classes += cur_classes
        # num_cur_classes = len(cur_classes)
        # num_pre_classes = len(pre_classes)
        # num_seen_classes = len(seen_classes)

        # Init sup data
        if sup_exp is not None:
            if tr_exp_idx > 0:
                # For subsequent tasks, we concat buffer and supervised dataloader
                sup_buff_exp = ConcatDataset([sup_exp, buffer_sup])
                sup_buff_exp.targets = torch.cat([torch.tensor(sup_exp.targets), torch.tensor(buffer_sup.labels)])
            else:
                sup_buff_exp = sup_exp
            sup_dataset = SupervisedDataset(data=sup_buff_exp, transforms=labeled_transforms, num_views=args.supervised_views)
            print(f'Len Supervised dataset (+ sup buffer): {len(sup_dataset)}')
            sampler = RandomSampler(sup_exp, replacement=True, num_samples=int(1e10))#TODO: use class balanced sampler (e.g. SimpleClassBalancedSampler in .data)
            sup_loader = DataLoader(sup_dataset, num_workers=args.num_workers, batch_size=args.sup_mb_size, sampler=sampler) 
        else:
            sup_loader = None

        # Init unsup current exp data
        if unsup_exp is not None:
            unsup_curr_dataset = UnsupervisedDataset(data=unsup_exp, transforms=unlabeled_transforms)
            print(f'Len Unsupervised Current dataset: {len(unsup_curr_dataset)}')
            unsup_curr_loader = DataLoader(unsup_curr_dataset, num_workers=args.num_workers, batch_size=args.unsup_mb_size, shuffle=True, drop_last=True)
            epochs = args.epochs
        else:
            epochs = args.last_exps_epochs
            unsup_curr_loader = None

        # Init unsup buffer data
        if tr_exp_idx > 0:
            unsup_buffer_dataset = UnsupervisedDataset(data=buffer_unsup, transforms=unlabeled_transforms)
            print(f'Len Unsupervised Buffer dataset: {len(unsup_buffer_dataset)}')
            sampler = RandomSampler(buffer_unsup, replacement=True, num_samples=int(1e10))
            unsup_buffer_loader = DataLoader(unsup_buffer_dataset, num_workers=args.num_workers, batch_size=args.unsup_buff_mb_size, sampler=sampler)
        else:
            unsup_buffer_loader = None

        # TODO: PROVVISORIO, NOT FOR LABEL DELAY
        if tr_exp_idx == 0:
            #  Calculate iterations per epoch (use unsup_curr_loader if available, otherwise use unsup_buffer_loader)
            if unsup_curr_loader is not None:
                ipe = ipe = len(unsup_curr_loader) # iterations per epoch
            else:
                # Iterating epochs over unsup_buffer_loader
                assert unsup_buffer_loader is not None, 'Both unsup_curr_loader and unsup_buffer_loader are None'
                ipe = len(unsup_buffer_loader)
                unsup_buffer_loader = DataLoader(unsup_buffer_dataset, num_workers=args.num_workers, batch_size=args.unsup_buff_mb_size, shuffle=True, drop_last=True)

            tot_steps = args.epochs * ipe * args.num_exps
            scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, tot_steps)

        # #  Calculate iterations per epoch (use unsup_curr_loader if available, otherwise use unsup_buffer_loader)
        # if unsup_curr_loader is not None:
        #     ipe = ipe = len(unsup_curr_loader) # iterations per epoch
        # else:
        #     # Iterating epochs over unsup_buffer_loader
        #     assert unsup_buffer_loader is not None, 'Both unsup_curr_loader and unsup_buffer_loader are None'
        #     ipe = len(unsup_buffer_loader)
        #     unsup_buffer_loader = DataLoader(unsup_buffer_dataset, num_workers=args.num_workers, batch_size=args.unsup_buff_mb_size, shuffle=True, drop_last=True)

        # exp_steps = args.epochs * ipe
        # scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, exp_steps)

        # Train

        for epoch in range(epochs):
            model.train()

            print(f'Doing epoch {epoch}')
            running_loss = 0.0
            running_loss_sup = 0.0
            running_loss_pseudo = 0.0
            running_loss_ssl = 0.0

            # Init iters
            if sup_exp is not None:
                iter_supervised = iter(sup_loader)
            if unsup_exp is not None:
                iter_unsup_curr = iter(unsup_curr_loader)
            if tr_exp_idx > 0:
                iter_unsup_buffer = iter(unsup_buffer_loader)

            # Iterate over dataloaders
            for iter_idx in tqdm(range(ipe)):

                if unsup_curr_loader is not None:
                    try:
                        u_curr_data = next(iter_unsup_curr)
                    except:
                        print('Exception: Empty unsupervised current dataloader')
                    finally:
                        [u_curr_x_weak, u_curr_x_strong], _ = u_curr_data
                        u_curr_x_weak  = u_curr_x_weak.to(device, non_blocking=True)
                        u_curr_x_strong = u_curr_x_strong.to(device, non_blocking=True)
                else:
                    u_curr_x_weak, u_curr_x_strong = None, None

                if sup_loader is not None:
                    try:
                        sdata = next(iter_supervised)
                    except:
                        print('Exception: Empty supervised dataloader')
                        print(f'len sup dataloader: {len(sup_loader)}')
                    finally:
                        [s_x1, s_x2], s_y = sdata
                        s_x1 = s_x1.to(device, non_blocking=True)
                        s_x2 = s_x2.to(device, non_blocking=True)
                        s_y = s_y.to(device, non_blocking=True)
                else:
                    s_x1, s_x2, s_y = None, None, None

                if unsup_buffer_loader is not None:
                    try:
                        u_buffer_data = next(iter_unsup_buffer)
                    except:
                        print('Exception: Empty unsupervised buffer dataloader')
                        print(f'len unsup buffer dataloader: {len(unsup_buffer_loader)}')
                    finally:
                        [u_buffer_x_weak, u_buffer_x_strong], _ = u_buffer_data
                        u_buffer_x_weak  = u_buffer_x_weak.to(device, non_blocking=True)
                        u_buffer_x_strong = u_buffer_x_strong.to(device, non_blocking=True)
                else:
                    u_buffer_x_weak, u_buffer_x_strong = None, None

                # Forward pass
                loss, loss_supervised, loss_pseudo, loss_ssl = model(s_x1=s_x1, s_x2=s_x2, s_y=s_y, 
                                                                     u_curr_x_weak=u_curr_x_weak, u_curr_x_strong=u_curr_x_strong,
                                                                     u_buff_x_weak=u_buffer_x_weak, u_buff_strong=u_buffer_x_strong)
                # print(f'loss: {loss}, loss_supervised: {loss_supervised}, loss_pseudo: {loss_pseudo}, loss_ssl: {loss_ssl}')
                
                running_loss += loss.item() if hasattr(loss, "item") else loss
                running_loss_sup += loss_supervised.item() if  hasattr(loss_supervised, "item") else loss_supervised
                running_loss_pseudo += loss_pseudo.item() if hasattr(loss_pseudo, "item") else loss_pseudo
                running_loss_ssl += loss_ssl.item() if hasattr(loss_ssl, "item") else loss_ssl
                
                loss = loss.mean()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Log losses
            with open(os.path.join(log_folder, 'loss.csv'), 'a') as f:
                f.write(f'{tr_exp_idx},{epoch},{running_loss_sup/ipe},{running_loss_pseudo/ipe},{running_loss_ssl/ipe},{running_loss/ipe}\n')

            if args.eval_every == 'epoch':
                # Eval
                evaluator.exec_eval(model=model, tr_exp_idx=tr_exp_idx, epoch=epoch)
        
        if args.eval_every == 'exp':
            # Eval
            evaluator.exec_eval(model=model, tr_exp_idx=tr_exp_idx, epoch=epoch)
        
        # Update buffers
        if sup_exp is not None:
            buffer_sup.update(sup_exp)
        if unsup_exp is not None:
            buffer_unsup.update(unsup_exp)

    # Checkpoint at the end of training
    chkpt_pth = os.path.join(log_folder, 'checkpoints')
    if not os.path.exists(chkpt_pth):
        os.makedirs(chkpt_pth)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }, os.path.join(chkpt_pth, f'checkpoint_trexp{tr_exp_idx}.pth'))