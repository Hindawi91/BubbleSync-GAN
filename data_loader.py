from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import numpy as np
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
        """Load Boiling dataset. Uses a recursive glob so images can sit
        either directly under train/test/<domain>/ or in further
        subfolders (e.g. class-specific subdirectories) without needing to
        reorganize your data -- both layouts are supported the same way."""

        # Load test dataset
        test_neg = glob(os.path.join(self.image_dir, 'test', 'domainA', '**', '*jpg'), recursive=True)
        test_pos = glob(os.path.join(self.image_dir, 'test', 'domainB', '**', '*jpg'), recursive=True)

        for filename in test_neg:
            self.test_dataset.append([filename, [0]])

        for filename in test_pos:
            self.test_dataset.append([filename, [1]])

        # Load train dataset
        train_neg = glob(os.path.join(self.image_dir, 'train', 'domainA', '**', '*jpg'), recursive=True)
        train_pos = glob(os.path.join(self.image_dir, 'train', 'domainB', '**', '*jpg'), recursive=True)

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


def _seed_worker(worker_id):
    """Give each DataLoader worker process a distinct, reproducible seed.

    Without this, worker processes inherit RNG state from the fork point and
    Python's `random` module (unlike torch's) is not auto-reseeded per
    worker, which can make augmentation (e.g. RandomHorizontalFlip) correlate
    across workers or vary between runs.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loader(image_dir, attr_path, selected_attrs, crop_size=128, image_size=128,
               batch_size=16, dataset='Boiling', mode='train', num_workers=1, seed=None):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    # transform.append(T.CenterCrop(crop_size))
    # Model expects 1-channel (grayscale) input -- without this conversion,
    # RGB/3-channel source images would mismatch the Generator/Discriminator's
    # expected input shape.
    transform.append(T.Grayscale(num_output_channels=1))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5,), std=(0.5,)))
    transform = T.Compose(transform)

    dataset = Boiling(image_dir, transform, mode)

    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = _seed_worker

    data_loader = data.DataLoader(dataset=dataset,
                                   batch_size=batch_size,
                                   shuffle=(mode == 'train'),
                                   num_workers=num_workers,
                                   worker_init_fn=worker_init_fn,
                                   generator=generator)
    return data_loader
