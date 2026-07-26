"""
SequenceSync-GAN-style data loader (triplet/sequential loading + temporal
discriminator), combined with BubbleSync-GAN, merging:
  - BubbleSync-GAN's proven patterns: recursive glob (handles DS1's flat
    layout and DS3's pre-CHF/post-CHF subfolder split with the same code
    path), grayscale conversion, seeded DataLoader workers for
    reproducibility.
  - SequenceSync-GAN's novelty: triplet (3-frame) sequential loading with
    an in-order/shuffled label for the temporal-discriminator loss.

Critical fix vs the original SequenceSync-GAN data_loader.py: the original
get_frame_no() used re.search(r'\\d+', filename), which grabs the FIRST
number in the filename -- wrong for both datasets here. Replaced with
get_temporal_sort_key(), which understands the actual per-dataset physical
ordering:

  DS1: "ONB_BIC<N>.jpg" (pre-CHF) then "CHF<N>.jpg" (post-CHF). CHF always
       comes after ONB_BIC regardless of N (the CHF frame counter resets),
       so sort key = (phase, N), phase=0 for ONB_BIC, phase=1 for CHF.

  DS3: "Boiling-5-<W>W-<N>.jpg" (pre-CHF, power ramps 10W..120W) then
       "Boiling-5_CHF-<N>.jpg" (post-CHF, no W value). Sort key =
       (phase, W, N), phase=0 for pre-CHF (sorted by W then N), phase=1 for
       post-CHF (W irrelevant, just N). Phase always dominates the sort, so
       every post-CHF frame sorts after every pre-CHF frame.

Confirmed against real HPC filenames and confirmed each split contains only
a single experimental run (only "Boiling-5" present across DS3
train/val/test) -- so no cross-run mixing concern within a split.
"""
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import torch
import os
import re
import random
import numpy as np
from glob import glob


def get_temporal_sort_key(filename):
    """Returns a (phase, secondary, frame_no) tuple such that sorting by
    this key produces the true chronological order for either DS1 or DS3
    filenames. Raises ValueError for anything unrecognized, rather than
    silently mis-ordering (that's exactly the class of bug this replaces)."""
    basename = os.path.basename(filename)

    # DS3 pre-CHF: Boiling-<run>-<W>W-<frame>.jpg
    m = re.match(r'.*-(\d+)W-(\d+)\.jpg$', basename, re.IGNORECASE)
    if m:
        w_val = int(m.group(1))
        frame_no = int(m.group(2))
        return (0, w_val, frame_no)

    # DS3 post-CHF: Boiling-<run>_CHF-<frame>.jpg
    m = re.match(r'.*_CHF-(\d+)\.jpg$', basename, re.IGNORECASE)
    if m:
        frame_no = int(m.group(1))
        return (1, 0, frame_no)

    # DS1 pre-CHF: ONB_BIC<frame>.jpg
    m = re.match(r'ONB_BIC(\d+)\.jpg$', basename, re.IGNORECASE)
    if m:
        frame_no = int(m.group(1))
        return (0, 0, frame_no)

    # DS1 post-CHF: CHF<frame>.jpg
    m = re.match(r'CHF(\d+)\.jpg$', basename, re.IGNORECASE)
    if m:
        frame_no = int(m.group(1))
        return (1, 0, frame_no)

    raise ValueError(f"Unrecognized filename pattern for temporal ordering: {filename}")


def _seed_worker(worker_id):
    """Seed each DataLoader worker distinctly, matching BubbleSync-GAN's
    reproducibility fix -- without this, Python's `random` module (used
    below for triplet index selection) is not auto-reseeded per worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Boiling(data.Dataset):
    """Dataset class for triplet/sequential loading from a single domain."""

    def __init__(self, image_dir, mode, domain_name, label, in_sequence=True, image_size=256):
        """
        image_dir: root folder containing one subfolder per dataset (DS1, DS2, DS3)
        domain_name: which dataset subfolder to load (e.g. 'DS3')
        label: 0 or 1, assigned by the caller based on source/target role
               (matches BubbleSync-GAN's convention: source_domain -> 0,
               target_domain -> 1)
        """
        self.image_dir = image_dir
        self.mode = mode
        self.domain_name = domain_name
        self.label = label
        self.in_sequence = in_sequence
        self.image_size = image_size
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.load_data()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        elif mode == 'val':
            self.num_images = len(self.val_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def load_data(self):
        """Load and temporally sort each split. Recursive glob handles both
        DS1's flat layout and DS3's pre-CHF/post-CHF subfolder split with
        the same code path (same trick as BubbleSync-GAN's data_loader)."""
        for split, dataset_list in (
            ('train', self.train_dataset),
            ('val', self.val_dataset),
            ('test', self.test_dataset),
        ):
            files = glob(os.path.join(self.image_dir, self.domain_name, split, '**', '*.jpg'), recursive=True)
            files_sorted = sorted(files, key=get_temporal_sort_key)
            for filename in files_sorted:
                dataset_list.append([filename, [self.label]])

        print(f'Finished loading {self.domain_name} ({self.mode} split sizes: '
              f'train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)})')

    def prepare_time_indices(self, index, data_len, in_sequence):
        """Unchanged from the original SequenceSync-GAN logic -- this part
        (picking 3 indices, in-order or shuffled) was not the buggy part;
        only the underlying sort order feeding into it was wrong."""
        if in_sequence:
            if index <= (data_len - 3):
                i = index
            else:
                i = index - 2
            j = random.randint(i + 1, data_len - 2)
            k = random.randint(j + 1, data_len - 1)
        else:
            excluded_numbers = []
            i = index
            excluded_numbers.append(i)
            j = random.randint(0, data_len - 1)
            while j in excluded_numbers:
                j = random.randint(0, data_len - 1)
            excluded_numbers.append(j)
            k = random.randint(0, data_len - 1)
            while k in excluded_numbers or k > j > i:
                k = random.randint(0, data_len - 1)
        return i, j, k

    def __getitem__(self, index):
        if self.mode == 'train':
            dataset = self.train_dataset
        elif self.mode == 'val':
            dataset = self.val_dataset
        else:
            dataset = self.test_dataset

        i, j, k = self.prepare_time_indices(index=index, data_len=len(dataset), in_sequence=self.in_sequence)

        transform = []
        if self.mode == 'train':
            transform.append(T.RandomHorizontalFlip())
        transform.append(T.Grayscale(num_output_channels=1))
        transform.append(T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.LANCZOS))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5,), std=(0.5,)))
        transform = T.Compose(transform)

        filename1, label1 = dataset[i]
        filename2, label2 = dataset[j]
        filename3, label3 = dataset[k]
        file_names = [filename1, filename2, filename3]

        # label1/2/3 are all identical (same domain), matches original design
        label = label1

        image1 = transform(Image.open(filename1))
        image2 = transform(Image.open(filename2))
        image3 = transform(Image.open(filename3))

        seq_img = torch.cat((image1, image2, image3), dim=0)
        seq_label = 1 if i < j < k else 0

        self.in_sequence = not self.in_sequence  # flip flag to balance in-order vs shuffled triplets

        return seq_img, torch.FloatTensor(label), seq_label, file_names

    def __len__(self):
        return self.num_images


def get_loader(image_dir, image_size=256, batch_size=16, mode='train', num_workers=1,
               domain_name='DS3', label=0, in_sequence=True, seed=None):
    dataset = Boiling(image_dir, mode, domain_name, label, in_sequence, image_size)

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
