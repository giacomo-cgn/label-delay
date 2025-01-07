import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Logging
    log_group = parser.add_argument_group('Logging', 'logging setup')
    log_group.add_argument('--log-root', type=str, default='./logs', help='log root')
    log_group.add_argument('--log-name', type=str, default='', help='log name')

    # Dataset
    data_group = parser.add_argument_group('Data', 'benchmark and dataset setup')
    data_group.add_argument('--dataset', type=str, default='cifar100', help='dataset')
    data_group.add_argument('--dataset-root', type=str, default='./data', help='dataset root')
    data_group.add_argument('--valid-ratio', type=float, default=0.1, help='valid ratio')
    data_group.add_argument('--num-exps', type=int, default=10, help='number of continual experiences')
    data_group.add_argument('--supervised-ratio', type=float, default=0.05, help='supervised samples ratio')
    data_group.add_argument('--unsup-anticipate-ratio', type=float, default=0.5, help='ratio of unsupervised samples anticipated to previous experience, creating label delay.\
                            1:all supervised samples are delayed, 0:no delay')
    data_group.add_argument('--drop-last-exp', action='store_true', help='drop the last experience. To be used with delay if you want uniform experiences.')
    data_group.add_argument('--dataset-seed', type=int, default=42, help='seed for dataset creation')

    # Model
    model_group = parser.add_argument_group('Model', 'model setup')
    model_group.add_argument('--model', type=str, default='nncsl', help='model')
    model_group.add_argument('--seed' , type=int, default=42, help='seed')
    model_group.add_argument('--backbone', type=str, default='resnet18', help='backbone')
    model_group.add_argument('--checkpoint', type=str, default=None, help='checkpoint')

    # Training
    training_group = parser.add_argument_group('Training', 'training setup')
    training_group.add_argument('--sup-mb-size', type=int, default=128, help='supervised batch size')
    training_group.add_argument('--sup-imgs-per-class', type=int, default=3, help='supervised images per class')
    training_group.add_argument('--unsup-mb-size', type=int, default=256, help='unsupervised batch size')
    training_group.add_argument('--num-workers', type=int, default=8, help='number of workers')
    training_group.add_argument('--epochs', type=int, default=250, help='number of epochs')
    training_group.add_argument('--gpu-idx', type=int, default=0, help='gpu index')

    # Optimizer
    optim_group = parser.add_argument_group('Optimizer', 'optimizer setup')
    optim_group.add_argument('--optimizer', type=str, default='lars', help='optimizer')
    optim_group.add_argument('--lr', type=float, default=1.2, help='learning rate')
    optim_group.add_argument('--lr-cls', type=float, default=0.12, help='learning rate for classifier')
    optim_group.add_argument('--start-lr', type=float, default=0.08, help='start learning rate')
    optim_group.add_argument('--final-lr', type=float, default=0.032, help='final learning rate')
    optim_group.add_argument('--nesterov' ,action='store_true', help='use nesterov momentum')
    optim_group.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    optim_group.add_argument('--scheduler', type=str, default='cosine', help='scheduler')
    optim_group.add_argument('--warmup', type=int, default=10, help='warmup epochs')

    # Buffer
    buffer_group = parser.add_argument_group('Buffer', 'buffer setup')
    buffer_group.add_argument('--buffer-size', type=int, default=500, help='buffer size')

    # NNCSL
    nncsl_group = parser.add_argument_group('NNCSL', 'nncsl setup')
    nncsl_group.add_argument('--label-smoothing', type=float, default=0.1, help='NNCSL label smoothing')
    nncsl_group.add_argument('--use-fp16', action='store_true', help='NNCSL use fp16')
    nncsl_group.add_argument('--output-dim', type=int, default=128, help='NNCSL output dimension')
    nncsl_group.add_argument('--w-paws', type=float, default=1.0, help='NNCSL PAWS weight')
    nncsl_group.add_argument('--w-me-max', type=float, default=1.0, help='NNCSL ME-MAX weight')
    nncsl_group.add_argument('--w-online', type=float, default=0.005, help='NNCSL online weight')
    nncsl_group.add_argument('--w-dist', type=float, default=0.2, help='NNCSL distillation weight')
    nncsl_group.add_argument('--alpha', type=float, default=0, help='NNCSL alpha')

    # Transforms
    transforms_group = parser.add_argument_group('Transforms', 'transformations setup')
    transforms_group.add_argument('--multicrop', type=int, default=2, help='number of smaller multiple crops per unsupervised image (additional to views)')
    transforms_group.add_argument('--supervised-views', type=int, default=2, help='number of supervised views')
    transforms_group.add_argument('--unsupervised-views', type=int, default=2, help='number of unsupervised views')
    transforms_group.add_argument('--normalize', action='store_true', help='normalize inputs')
    transforms_group.add_argument('--color-jitter-strength', type=float, default=0.5, help='color jitter strength')
    
    

    parse_args = parser.parse_args()

    return parse_args