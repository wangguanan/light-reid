import torch
import torchvision
from torchvision import transforms
from .dataset import PersonReIDDataSet

class CustomedLoaders:
    '''
    load customed dataset
    query_path and gallery_path should have the following structure
    |____ data_path/
         |____ person_id_1/
              |____ 1.jpg
              |____ 2.jpg
              ......
         |____ person_id_2/
         |____ person_id_2/
         ......
    '''

    def __init__(self, config):

        self.config = config
        self.query_path = config.query_path
        self.gallery_path = config.gallery_path
        self.transform_test = transforms.Compose([
            transforms.Resize(config.image_size, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        query_dataset = torchvision.datasets.ImageFolder(self.query_path)
        gallery_dataset = torchvision.datasets.ImageFolder(self.gallery_path)
        self.query_samples = [list(sample)+[1] for sample in query_dataset.samples] # set all camera id as 1
        self.gallery_samples = [list(sample)+[1] for sample in gallery_dataset.samples] # set all camera id as 1
        self.query_dataset = PersonReIDDataSet(self.query_samples, self.transform_test)
        self.gallery_dataset = PersonReIDDataSet(self.gallery_samples, self.transform_test)

        self.query_loader = \
            torch.utils.data.DataLoader(self.query_dataset, batch_size=64, num_workers=8, drop_last=False, shuffle=False)
        self.gallery_loader = \
            torch.utils.data.DataLoader(self.gallery_dataset, batch_size=64, num_workers=8, drop_last=False, shuffle=False)
