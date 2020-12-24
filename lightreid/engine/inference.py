import torch
import numpy as np
from PIL import Image
from collections import OrderedDict
from easydict import EasyDict as edict

from .build import ENGINEs_REGISTRY


@ENGINEs_REGISTRY.register()
class Inference:
    '''
    A class for inference only
    Args:
        model(lightreid.ReIDModel)
        img_size(tuple): height, width
        model_path(str): path to model_state_dict.pth
        use_gpu(bool): use gpu to extract features
        light_feat(bool): if True, the model output is binary code [0,1]
    ExamplesL:

    '''

    def __init__(self, model, img_size, model_path, use_gpu=False, light_feat=False, **kwargs):
        '''
        Args:
            model(lightreid.BaseReIDModel)
            img_size(tuple): height, width
            use_gpu(bool): True or False
        '''
        self.height, self.width = img_size
        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')

        mean = [0.485, 0.456, 0.406] if 'mean' not in kwargs.keys() else kwargs['mean']
        std = [0.229, 0.224, 0.225] if 'std' not in kwargs.keys() else kwargs['std']
        self.set_mean(mean=mean)
        self.set_std(std=std)

        # init
        model = self.resume_from_path(model, model_path)
        if light_feat:
            print('enable tanh, output binary code')
            model.enable_tanh()
        self.model = model.to(self.device).eval()


    def set_mean(self, mean):
        print('set mean {}'.format(mean))
        self.mean = mean

    def set_std(self, std):
        print('set std {}'.format(std))
        self.std = std


    def process(self, inputs, return_type='numpy'):
        '''
        Args:
            inputs: accept
                torch.tensor: image(s) of size [(num), 3, h, w] normalized by ImageNet mean and std, RGB format
                np.ndarray: image(s) of size [(num), 3, h, w] in range [0, 255], RGB format
                str: an image path
                list(str): a list of image paths
            return_type(str): accept
                'numpy': np.array of size [num, dim]
                'torch': torch.tensor of size [num, dim]
        '''
        imgs = self.process_inputs(inputs)
        with torch.no_grad():
            feats = self.model(imgs)

        if return_type == 'numpy':
            return feats.data.cpu().numpy()
        elif return_type == 'torch':
            return feats.data


    def process_inputs(self, inputs):
        '''
        process inputs to formal format as below:
        Args:
            inputs: accept
                torch.tensor: image(s) of size [(num), 3, h, w] normalized by ImageNet mean and std
                np.ndarray: image(s) of size [(num), 3, h, w] in range [0, 255]
                str: an image path
                list(str): a list of image paths
        Return:
            outputs(torch.tensor): [num, 3, h, w], normalized with ImageNet mean and std in RGB
        '''

        # if inputs are np.ndarray of size [3,h,w] or [num,3,h,w]
        if isinstance(inputs, np.ndarray):
            if len(inputs.shape) == 3:
                outputs = np.expand_dims(inputs, dim=0)
            elif len(inputs.shape) == 4:
                outputs = inputs
            else:
                raise RuntimeError('inputs dimension error, expect 3([3, h, w]) or 4(num, 3, h, w), but got {}'.format(len(inputs.shape)))

        # if inputs are torch.tensor
        elif isinstance(inputs, torch.Tensor):
            if len(inputs.shape) == 3:
                outputs = inputs.unsqueeze(0)
            elif len(inputs.shape) == 4:
                outputs = inputs
            else:
                raise RuntimeError('inputs dimension error, expect 3([3, h, w]) or 4(num, 3, h, w), but got {}'.format(len(inputs.shape)))
            outputs = outputs.to(self.device)
            return outputs

        # if inputs is path(s)
        elif isinstance(inputs, str): # if input a path
            img = Image.open(inputs)  # load img as [0, 255], HWC, RGB
            img = img.resize([self.width, self.height], resample=2)  # resize
            img_numpy = np.array(img)  # astype numpy
            outputs = np.expand_dims(img_numpy, axis=0)
            outputs = outputs.transpose([0, 3, 1, 2])  # [num, h, w, 3] --> [num, 3, h, w]
        elif isinstance(inputs, list): # if input a list of path
            outputs = []
            for input in inputs:
                img = Image.open(input) # load img as [0, 255], HWC, RGB
                img = img.resize([self.width, self.height], resample=2) # resize
                img_numpy = np.array(img) # astype numpy
                outputs.append(img_numpy)
            outputs = np.asarray(outputs)
            outputs = outputs.transpose([0, 3, 1, 2])  # [num, h, w, 3] --> [num, 3, h, w]

        outputs = outputs / 255. # [0, 255] --> [0, 1]
        # normalize with Imagenet mean and std
        mean = np.array(self.mean).reshape([1, -1, 1, 1])
        std = np.array(self.std).reshape([1, -1, 1, 1])
        outputs = (outputs - mean) / std

        outputs = torch.from_numpy(outputs).float().to(self.device)
        return outputs


    def resume_from_path(self, model, model_path):
        '''resume from model. model_path shoule be like /path/to/model.pkl'''
        # self.model.load_state_dict(torch.load(model_path), strict=False)
        # print(('successfully resume model from {}'.format(model_path)))
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        new_state_dict = OrderedDict()
        matched_layers, discarded_layers = [], []
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # discard module.
            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)
        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)
        if len(discarded_layers) > 0:
            print('discarded layers: {}'.format(discarded_layers))
        return model
