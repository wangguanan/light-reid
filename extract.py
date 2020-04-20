'''
this file provide operation of extracting
a image, a batch of image, a list of image
and return its/their features
'''
import torch
import argparse
from core import Extractor


parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128], help='should be consistent with pre-trained model')
parser.add_argument('--pid_num', type=int, default=751, help='751 for Market-1501, 702(maybe 751) for DukeMTMC-reID')
parser.add_argument('--model_path', type=str, default='751', help='pre-trained model path')
config = parser.parse_args()


ReIDExtractor = Extractor(image_size=config.image_size, pid_num=config.pid_num, model_path=config.model_path)

image = torch.rand([3, 22, 54])
feature = ReIDExtractor.extract_image(image)

images = torch.rand([3, 3, 22, 54])
features = ReIDExtractor.extract_images(images)

image_list = [torch.rand([3, 22, 54]), torch.rand([3, 22, 54]), torch.rand([3, 22, 54])]
feature_list = ReIDExtractor.extract_list(images)
