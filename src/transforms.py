import torch
from torchvision import transforms

from PIL import ImageOps, ImageFilter

def make_transforms(
    dataset_name,
    training=True,
    basic_augmentations=False,
    force_center_crop=False,
    crop_scale=(0.08, 1.0),
    color_jitter=1.0,
    normalize=False,
):
    """
    :param dataset_name: ['imagenet', 'cifar10']
    :param training: whether the data is for training or evaluation
    :param basic_augmentations: whether to use simple data-augmentations
    :param force_center_crop: whether to force use of a center-crop
    :param color_jitter: strength of color-jitter
    :param normalize: whether to normalize color channels
    """

    if 'imagenet' in dataset_name:

        return _make_imgnt_transforms(
            training=training,
            basic=basic_augmentations,
            force_center_crop=force_center_crop,
            normalize=normalize,
            color_distortion=color_jitter,
            scale=crop_scale)

    elif 'cifar' in dataset_name:

        return _make_cifar_transforms(
            training=training,
            basic=basic_augmentations,
            force_center_crop=force_center_crop,
            normalize=normalize,
            scale=crop_scale,
            color_distortion=color_jitter)


def _make_cifar_transforms(
    training=True,
    basic=False,
    force_center_crop=False,
    normalize=False,
    scale=(0.5, 1.0),
    color_distortion=0.5,
):
    """
    Make data transformations

    :param training: generate data transforms for train (alternativly test)
    :param basic: whether train transforms include more sofisticated transforms
    :param force_center_crop: whether to override settings and apply center crop to image
    :param normalize: whether to normalize image means and stds
    :param scale: random scaling range for image before resizing
    :param color_distortion: strength of color distortion
    """
    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)

        color_distort = transforms.Compose([
            rnd_color_jitter,
            Solarize(p=0.2),
            Equalize(p=0.2)])
        return color_distort

    if training and (not force_center_crop):
        if basic:
            transform = transforms.Compose(
                [transforms.CenterCrop(size=32),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor()])
        else:
            transform = transforms.Compose(
                [transforms.RandomResizedCrop(size=32, scale=scale),
                 transforms.RandomHorizontalFlip(),
                 get_color_distortion(s=color_distortion),
                 transforms.ToTensor()])
    else:
        transform = transforms.Compose(
            [transforms.CenterCrop(size=32),
             transforms.ToTensor()])

    if normalize:
        transform = transforms.Compose(
            [transform,
             transforms.Normalize(
                 (0.4914, 0.4822, 0.4465),
                 (0.2023, 0.1994, 0.2010))])

    return transform


def _make_imgnt_transforms(
    training=True,
    basic=False,
    force_center_crop=False,
    normalize=False,
    scale=(0.08, 1.0),
    color_distortion=1.0,
):
    """
    Make data transformations

    :param training: generate data transforms for train (alternativly test)
    :param basic: whether train transforms include more sofisticated transforms
    :param force_center_crop: whether to override settings and apply center crop to image
    :param normalize: whether to normalize image means and stds
    :param scale: random scaling range for image before resizing
    :param color_distortion: strength of color distortion
    """
    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    if training and (not force_center_crop):
        if basic:
            transform = transforms.Compose(
                [transforms.RandomResizedCrop(size=224, scale=scale),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor()])
        else:
            transform = transforms.Compose(
                [transforms.RandomResizedCrop(size=224, scale=scale),
                 transforms.RandomHorizontalFlip(),
                 get_color_distortion(s=color_distortion),
                 GaussianBlur(p=0.5),
                 transforms.ToTensor()])
    else:
        transform = transforms.Compose(
            [transforms.Resize(size=256),
             transforms.CenterCrop(size=224),
             transforms.ToTensor()])

    if normalize:
        transform = transforms.Compose(
            [transform,
             transforms.Normalize(
                 (0.485, 0.456, 0.406),
                 (0.229, 0.224, 0.225))])

    return transform



def make_multicrop_transform(
    dataset_name,
    size,
    crop_scale,
    normalize,
    color_distortion
):
    if 'imagenet' in dataset_name:
        return _make_multicrop_imgnt_transforms(
            size=size,
            scale=crop_scale,
            normalize=normalize,
            color_distortion=color_distortion)
    elif 'cifar10' in dataset_name:
        return _make_multicrop_cifar10_transforms(
            size=size,
            scale=crop_scale,
            normalize=normalize,
            color_distortion=color_distortion)

def _make_multicrop_cifar10_transforms(
    size=18,
    scale=(0.3, 0.75),
    normalize=False,
    color_distortion=0.5
):

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)

        color_distort = transforms.Compose([
            rnd_color_jitter,
            Solarize(p=0.2),
            Equalize(p=0.2)])
        return color_distort

    transform = transforms.Compose(
        [transforms.RandomResizedCrop(size=size, scale=scale),
         transforms.RandomHorizontalFlip(),
         get_color_distortion(s=color_distortion),
         transforms.ToTensor()])

    if normalize:
        transform = transforms.Compose(
            [transform,
             transforms.Normalize(
                 (0.4914, 0.4822, 0.4465),
                 (0.2023, 0.1994, 0.2010))])

    return transform


def _make_multicrop_imgnt_transforms(
    size=96,
    scale=(0.05, 0.14),
    normalize=False,
    color_distortion=1.0,
):
    def get_color_distortion(s=1.0):
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    transform = transforms.Compose(
        [transforms.RandomResizedCrop(size=size, scale=scale),
         transforms.RandomHorizontalFlip(),
         get_color_distortion(s=color_distortion),
         GaussianBlur(p=0.5),
         transforms.ToTensor()])

    if normalize:
        transform = transforms.Compose(
            [transform,
             transforms.Normalize(
                 (0.485, 0.456, 0.406),
                 (0.229, 0.224, 0.225))])

    return transform


class Solarize(object):
    def __init__(self, p=0.2):
        self.prob = p

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        v = torch.rand(1) * 256
        return ImageOps.solarize(img, v)


class Equalize(object):
    def __init__(self, p=0.2):
        self.prob = p

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        return ImageOps.equalize(img)


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))