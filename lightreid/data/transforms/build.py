from .random_erasing import RandomErasing
from .pad_crop import padcrop
from .autoaug import ImageNetPolicy
from .augmix import AugMix
from .half_crop import HalfCrop
import torchvision.transforms as transforms


__transforms_factory_before = {
    'autoaug': ImageNetPolicy(prob=0.5),
    'randomflip': transforms.RandomHorizontalFlip(p=0.5),
    'padcrop': padcrop,
    'colorjitor': transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0),
    'augmix': AugMix(prob=0.5),
    'halfcrop': HalfCrop(prob=0.5, keep_range=[0.5, 1.5])
}

__transforms_factory_after = {
    'rea': RandomErasing(probability=0.5)
}

__KWARGS = ['total_epochs', 'mean', 'std']

def build_transforms(img_size, transforms_list, **kwargs):

    for transform in transforms_list:
        assert transform in __transforms_factory_before.keys() or transform in __transforms_factory_after.keys(), \
            'Expect transforms in {} and {}, got {}'.format(__transforms_factory_before.keys(), __transforms_factory_after.keys(), transform)

    # for key in kwargs.keys():
    #     assert key in __KWARGS, 'expect parameter in {} but got {}'.format(__KWARGS, key)

    results = [transforms.Resize(img_size, interpolation=3)]
    for transform in transforms_list:
        if transform in __transforms_factory_before.keys():
            if transform == 'padcrop':
                results.append(__transforms_factory_before[transform](img_size))
            else:
                results.append(__transforms_factory_before[transform])

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if 'mean' in kwargs.keys():
        mean = kwargs['mean']
        print('set mean {}'.format(mean))
    if 'std' in kwargs.keys():
        std = kwargs['std']
        print('set std {}'.format(std))
    results.extend(# totensor --> normalize
        [transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std)])

    for transform in transforms_list:
        if transform in __transforms_factory_after.keys():
            results.append(__transforms_factory_after[transform])

    return transforms.Compose(results)





