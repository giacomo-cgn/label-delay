import torch
from torchvision.models import resnet18, resnet34, resnet50

def get_encoder(encoder_name: str = 'resnet18',
                dataset: str = 'cifar100'
                ):
    
   if encoder_name == 'resnet18':
      encoder = resnet18(pretrained=False)
   elif encoder_name == 'resnet34':
      encoder = resnet34(pretrained=False)
   elif encoder_name == 'resnet50':
      encoder = resnet50(pretrained=False)
   else:
      raise ValueError('Unknown encoder name')
   
   # Reduce dimension of the first ResNet layer for CIFAR
   if 'cifar' in dataset:
      encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
      encoder.maxpool = torch.nn.Identity()

   encoder.features_dim = encoder.fc.weight.shape[1]
   # Remove the last layer
   encoder.fc = torch.nn.Identity()
    
   return encoder
    