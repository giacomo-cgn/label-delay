
from tqdm import tqdm
import copy
import datetime
import os

import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F

from ..data import get_benchmark_label_delay, ClassStratifiedSampler, SupervisedDataset, UnsupervisedDataset
from ..model import init_model
from ..loss import init_paws_loss
from ..buffer import ClassBalancedBuffer
from ..utils import make_labels_matrix, WarmupCosineSchedule
from ..optimizer import init_opt
from ..transforms import make_multicrop_transform, make_transforms
from ..evaluate import exec_eval



def train_nncsl(args, log_folder, device):
    mask = False

    # transformations
    if 'cifar' in args.dataset or 'tiny' in args.dataset:
        crop_scale = (0.75, 1.0) if args.multicrop > 0 else (0.5, 1.0)
        mc_scale = (0.3, 0.75)
        mc_size = 18
    else:
        crop_scale = (0.14, 1.0) if args.multicrop > 0 else (0.08, 1.0)
        mc_scale = (0.05, 0.14)
        mc_size = 96
    
    if args.multicrop > 0:
        multicrop_transforms = make_multicrop_transform(dataset_name=args.dataset,
                                              size=mc_size, crop_scale=mc_scale, normalize=args.normalize,
                                              color_distortion=args.color_jitter_strength)
        
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
    

    # Initialize model
    encoder = init_model(
        device=device,
        model_name=args.backbone,
        use_pred=False,
        output_dim=args.output_dim,
        cifar='cifar' in args.dataset,
        num_classes=benchmark.num_classes,
        detach=True)

    # Initialize optimizer
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16)
    encoder, optimizer = init_opt(
        optimizer_name=args.optimizer,
        encoder=encoder,
        weight_decay=args.weight_decay,
        ref_lr=args.lr,
        nesterov=args.nesterov,
        ref_lr_cls=args.lr_cls)
    
    # Initialize losses
    paws, snn, sharpen = init_paws_loss(
        multicrop=2,
        tau=0.1,
        T=0.25,
        me_max=True)


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

        if buffer.__len__ == 0:
            num_classes_cl = num_cur_classes
        else:
            if mask:
                num_classes_cl = num_cur_classes
            else:
                num_classes_cl = num_seen_classes

        # Init dataloaders
        if tr_exp_idx > 0:
            # For subsequent tasks, we concat buffer and supervised dataloader
            sup_buff_exp = ConcatDataset([sup_exp, buffer])
            sup_buff_exp.targets = torch.cat([torch.tensor(sup_exp.targets), torch.tensor(buffer.labels)])
        else:
            sup_buff_exp = sup_exp

        # Add custom datasets with transformations
        sup_dataset = SupervisedDataset(data=sup_buff_exp, transforms=transforms, num_views=args.supervised_views)
        unsup_dataset = UnsupervisedDataset(data=unsup_exp, transforms=transforms, num_views=args.unsupervised_views,
                                            crop_transforms=multicrop_transforms, num_crop=args.multicrop)
        print('LEN UNSUP DATASET:', len(unsup_dataset))

        print('LEN SUP DATASET:', len(sup_dataset))

        sampler = ClassStratifiedSampler(data_source=sup_buff_exp, batch_size=args.sup_imgs_per_class, classes_per_batch=num_seen_classes, epochs=args.epochs)
       
        sup_loader = DataLoader(sup_dataset, num_workers=args.num_workers, batch_sampler=sampler)
        unsup_loader = DataLoader(unsup_dataset, batch_size=args.unsup_mb_size, shuffle=True, num_workers=args.num_workers, drop_last=True)


        ipe = len(unsup_loader) # iterations per epoch
        #  Init scheduler
        scheduler = WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=args.warmup*ipe,
            start_lr=args.start_lr,
            ref_lr=args.lr,
            T_max=args.epochs*ipe,
            final_lr=args.final_lr)

        if tr_exp_idx > 0:
            # Frozen last task encoder
            pre_encoder = copy.deepcopy(encoder.eval())

            if buffer.__len__ == 0:
                pre_labels_matrix = make_labels_matrix(
                    num_classes=num_cur_classes,
                    s_batch_size=args.sup_imgs_per_class,
                    device=device,
                    smoothing=args.label_smoothing,
                    task_idx=tr_exp_idx)
            else:
                pre_labels_matrix = make_labels_matrix(
                    num_classes=num_pre_classes,
                    s_batch_size=args.sup_imgs_per_class,
                    device=device,
                    smoothing=args.label_smoothing,
                    task_idx=tr_exp_idx)
        else:
            pre_encoder = None

        labels_matrix = make_labels_matrix(
            num_classes=num_classes_cl,
            s_batch_size=args.sup_imgs_per_class,
            device=device,
            smoothing=args.label_smoothing,
            task_idx=tr_exp_idx)


        # Train
        encoder.train()
        for epoch in range(args.epochs):
            print(f'Doing epoch {epoch}')

            # Iterate over dataloaders
            iter_supervised = iter(sup_loader)
            for udata in tqdm(unsup_loader):               

                # -- unsupervised imgs
                uimgs = [u.to(device, non_blocking=True) for u in udata[0]] + [u.to(device, non_blocking=True) for u in udata[1]]
                # print('uimgs[0] shape:', uimgs[0].shape)
                
                try:
                    sdata = next(iter_supervised)
                except:
                    print('Exception: Empty supervised dataloader')
                finally:
                    slabels = sdata[1].to(device, non_blocking=True).repeat(args.supervised_views)
                    # print(slabels)
                    # print(f'slabels.shape: {slabels.shape}')
                    plabels = torch.cat([labels_matrix for _ in range(args.supervised_views)])   
                    simgs = [s.to(device, non_blocking=True) for s in sdata[0]]
                    # print('simgs shape:', simgs[0].shape)

                # -- concatenate supervised imgs and unsupervised imgs
                imgs = simgs + uimgs


                with torch.cuda.amp.autocast(enabled=args.use_fp16):
                    optimizer.zero_grad()
                    # --
                    # h: representations of 'imgs' before head
                    # z: representations of 'imgs' after head
                    # -- If use_pred_head=False, then encoder.pred (prediction
                    #    head) is None, and _forward_head just returns the
                    #    identity, z=h
                    h, z, l = encoder(imgs, return_before_head=True)
                    if pre_encoder is not None:
                        with torch.no_grad():
                            pre_h, pre_z, pre_l = pre_encoder(imgs, return_before_head=True)
                        h_proj = encoder.feat_proj(h)

                    # Compute paws loss in full precision
                    with torch.cuda.amp.autocast(enabled=False):

                        # Step 1. convert representations to fp32
                        h, z, l = h.float(), z.float(), l.float()
                        if pre_encoder is not None:
                            pre_h, pre_z, pre_l = pre_h.float(), pre_z.float(), pre_l.float()
                            h_proj = h_proj.float()
                        

                        # Step 2. determine anchor views/supports and their
                        #         corresponding target views/supports
                        # --
                        if buffer.__len__ == 0:
                            num_support_mix = (args.supervised_views) * args.sup_imgs_per_class * num_cur_classes
                        else:
                            num_support_mix = (args.supervised_views) * args.sup_imgs_per_class * num_seen_classes
                        num_u_data_mix =  args.unsup_mb_size 
                        # --
                        if mask:
                            labels_mask = [label in cur_classes for label in slabels]
                        else:
                            labels_mask = [label in seen_classes for label in slabels]

                        plabels_masked= plabels
                        anchor_supports = z[:num_support_mix][labels_mask]
                        anchor_views = z[num_support_mix:]
                        # --
                        target_supports = h[:num_support_mix].detach()[labels_mask]
                        target_views = h[num_support_mix:].detach()
                        target_views = torch.cat([
                            target_views[num_u_data_mix:2*num_u_data_mix],
                            target_views[:num_u_data_mix]], dim=0)

                        # Step 3. compute paws loss with me-max regularization
                        (ploss, me_max, probs_anchor) = paws(
                            anchor_views=anchor_views,
                            anchor_supports=anchor_supports,
                            anchor_support_labels=plabels_masked,
                            target_views=target_views,  
                            target_supports=target_supports,
                            target_support_labels=plabels_masked,
                            mask=labels_mask)

                        # Step 4. compute online eval loss
                        # slogits = l[:num_support_mix,:num_seen_classes]
                        slogits = l[:num_support_mix]
                        # # # Change the targets to onehot label for mix up
                        # online_eval_loss = cross_entropy_with_logits(slogits, olabels)
                        
                        online_eval_loss = F.cross_entropy(slogits, slabels)


                        # Step 5. Distillation
                        dist_loss = torch.tensor(0)
                        dist_logit_loss = torch.tensor(0)
                        if pre_encoder is not None:
                            mse_loss = torch.nn.MSELoss()     
                            sigmoid = torch.nn.Sigmoid() 
                            softmax = torch.nn.Softmax(dim=1)  
                            cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                            if buffer.__len__ == 0:
                                pre_mask = [True for label in slabels]
                            else:
                                pre_mask = [label in pre_classes for label in slabels]
                            
                            # Distillation on features
                            dist_loss = mse_loss(h_proj, pre_h)

                            # Distillation on pseudo labels
                            # plabels for distillation with pre labeled data
                            pre_plabels = torch.cat([pre_labels_matrix for _ in range(args.supervised_views)])
                            cur_anchor_supports = z[:num_support_mix][pre_mask]
                            cur_anchor_views = z[num_support_mix:]
                            pre_anchor_supports = pre_z[:num_support_mix][pre_mask]
                            pre_anchor_views = pre_z[num_support_mix:]
                            pre_target_views = pre_z[num_support_mix:].detach()
                            pre_target_views = torch.cat([pre_target_views[num_u_data_mix:2*num_u_data_mix],
                                                            pre_target_views[:num_u_data_mix]], dim=0)

                            # Distillation with anchor views (snn)
                            cur_probs = snn(cur_anchor_views, cur_anchor_supports,  pre_plabels)
                            pre_probs = snn(pre_anchor_views, pre_anchor_supports,  pre_plabels)
                            dist_logit_loss = torch.mean(torch.sum(torch.log(cur_probs**(-pre_probs)), dim=1))
    
                    loss = args.w_paws*ploss + args.w_me_max*me_max  + args.w_online*online_eval_loss +\
                        args.w_dist*(args.alpha*dist_loss + (1-args.alpha)*dist_logit_loss)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

        # Update buffer after each experience
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
                  val_stream=benchmark.valid_stream, device=device, log_folder=log_folder, torch_resnet=False)
        
