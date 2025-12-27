from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
from glob import glob


class Boiling(data.Dataset):
    """Dataset class for the Boiling dataset."""

    def __init__(self, image_dir, transform, mode):
        """Initialize and Load the Boiling dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.load_data()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def load_data(self):
        """Load Boiling dataset"""
        
        # Load test dataset
        test_neg = glob(os.path.join(self.image_dir, 'test', 'domainA', '*jpg'))
        test_pos = glob(os.path.join(self.image_dir, 'test', 'domainB', '*jpg'))

        for filename in test_neg:
            self.test_dataset.append([filename, [0]])

        for filename in test_pos:
            self.test_dataset.append([filename, [1]])


        # Load train dataset
        train_neg = glob(os.path.join(self.image_dir, 'train', 'domainA', '*jpg'))
        train_pos = glob(os.path.join(self.image_dir, 'train', 'domainB', '*jpg'))

        for filename in train_neg:
            self.train_dataset.append([filename, [0]])

        for filename in train_pos:
            self.train_dataset.append([filename, [1]])

        print('Finished loading the Boiling dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(filename)
        if self.mode == 'train':
            return self.transform(image), torch.FloatTensor(label)
        else:
            return self.transform(image), torch.FloatTensor(label), filename

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=128, image_size=128, 
               batch_size=16, dataset='Boiling', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    #transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5,), std=(0.5,)))
    # transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = Boiling(image_dir, transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader
