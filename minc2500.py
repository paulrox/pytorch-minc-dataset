from __future__ import print_function
import os
from PIL import Image
from torch.utils.data import Dataset


class MINC2500(Dataset):
    def __init__(self, root_dir, set_type, split, transform=None):
        self.root_dir = root_dir
        self.set_type = set_type
        self.image_dir = os.path.join(self.root_dir, 'JPEGImages')
        self.image_sets = os.path.join(self.root_dir, 'ImageSets', 'Main')
        self.transform = transform

        # Get the material categories from the categories.txt file
        file_name = os.path.join(root_dir, 'categories.txt')
        self.categories = {}
        with open(file_name, 'r') as f:
            for i, line in enumerate(f):
                # The last line char (\n) must be removed
                self.categories[line[:-1]] = i

        # Load the image paths
        self.data = []
        file_name = os.path.join(root_dir, 'labels')
        # For the moment I use only the first split
        file_name = os.path.join(file_name, set_type + str(split) + '.txt')
        with open(file_name, 'r') as f:
            for line in f:
                img_path = line.split(os.sep)
                # The last line char (\n) must be removed
                self.data.append([line[:-1], self.categories[img_path[1]]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data[idx][0])
        image = Image.open(img_path)
        # Sometimes the images are opened as grayscale, so I need to force RGB
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, self.data[idx][1]
