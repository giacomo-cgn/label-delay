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
    # data_group.add_argument('--unsup-anticipate-ratio', type=float, default=0.5, help='ratio of unsupervised samples anticipated to previous experience, creating label delay.\
    #                         1:all supervised samples are delayed, 0:no delay')
    # data_group.add_argument('--drop-last-exp', action='store_true', help='drop the last experience. To be used with delay if you want uniform experiences.')
    data_group.add_argument('--delay', type=int, default=1, help='delay of the supervised data. 1:delay of one experience, 2:delay of two experiences, etc.')
    data_group.add_argument('--dataset-seed', type=int, default=42, help='seed for dataset creation')

    # Model
    model_group = parser.add_argument_group('Model', 'model setup')
    model_group.add_argument('--model', type=str, default='nncsl', help='model')
    model_group.add_argument('--seed' , type=int, default=42, help='seed')
    model_group.add_argument('--encoder', type=str, default='resnet18', help='encoder')

    # Hybrid SSL semi-supervised
    model_group.add_argument('--proj-hidden-dim', type=int, default=2048, help='projection hidden dimension')
    model_group.add_argument('--proj-output-dim', type=int, default=2048, help='projection output dimension')
    model_group.add_argument('--adapter-hidden-dim', type=int, default=2048, help='adapter hidden dimension')
    model_group.add_argument('--lambda-barlow', type=float, default=5e-3, help='lambda barlow')
    model_group.add_argument('--pseudo-label-temp', type=float, default=1.0, help='pseudo label temperature')
    model_group.add_argument('--pseudo-label-thresh', type=float, default=0.95, help='pseudo label threshold')
    model_group.add_argument('--omega-pseudo', type=float, default=1.0, help='omega pseudo')
    model_group.add_argument('--omega-ssl', type=float, default=1.0, help='omega ssl')

    # Training
    training_group = parser.add_argument_group('Training', 'training setup')
    training_group.add_argument('--sup-mb-size', type=int, default=64, help='supervised batch size')
    training_group.add_argument('--unsup-mb-size', type=int, default=256, help='unsupervised batch size')
    training_group.add_argument('--unsup-buff-mb-size', type=int, default=256, help='unsupervised buffer batch size')
    training_group.add_argument('--num-workers', type=int, default=8, help='number of workers')
    training_group.add_argument('--epochs', type=int, default=100, help='number of epochs')
    training_group.add_argument('--gpu-idx', type=int, default=0, help='gpu index')
    training_group.add_argument('--last-exps-epochs', type=int, default=25, help='how many epochs to use for the last exps\
                                 without unsupervised data, iterating over the unsupervised buffer')

    # Optimizer
    optim_group = parser.add_argument_group('Optimizer', 'optimizer setup')
    optim_group.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    optim_group.add_argument('--lr', type=float, default=0.03, help='learning rate')
    optim_group.add_argument('--lr-cls', type=float, default=0.01, help='learning rate for classifier') # 0.12
    optim_group.add_argument('--nesterov' ,action='store_true', help='use nesterov momentum', default=True)
    optim_group.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    optim_group.add_argument('--warmup', type=int, default=0, help='warmup epochs') # 5

    # Buffer
    buffer_group = parser.add_argument_group('Buffer', 'buffer setup')
    buffer_group.add_argument('--u-buffer-size', type=int, default=2000, help='buffer size for unsupervised data')
    buffer_group.add_argument('--s-buffer-size', type=int, default=500, help='buffer size for supervised data')


    # Transforms
    transforms_group = parser.add_argument_group('Transforms', 'transformations setup')
    transforms_group.add_argument('--supervised-views', type=int, default=2, help='number of supervised views') 
    transforms_group.add_argument('--use-barlow-transforms', action='store_true', default=False, help='use barlow transforms for strong augmentation')

    # Evaluation
    eval_group = parser.add_argument_group('Evaluation', 'evaluation setup')
    eval_group.add_argument('--eval-every', type=str, default='epoch', help='evaluate every "epoch" or "exp" (experience)')
    

    parse_args = parser.parse_args()

    return parse_args