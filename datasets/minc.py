from __future__ import print_function
import os
from PIL import Image
from torch import Tensor, is_tensor
from torch.utils.data import Dataset
from torchvision import transforms


class MINC(Dataset):
    def __init__(self, root_dir, set_type, scale=0.233, transform=None):
        self.root_dir = root_dir
        self.set_type = set_type
        self.transform = transform
        self.scale = scale
        # Those values are computed using the script 'get_minc2500_norm.py'
        self.mean = Tensor([0.507207, 0.458292, 0.404162])
        self.std = Tensor([0.254254, 0.252448, 0.266003])

        # Get the material categories from the categories.txt file
        file_name = os.path.join(root_dir, 'categories.txt')
        self.categories = {}
        with open(file_name, 'r') as f:
            for i, line in enumerate(f):
                # The last line char (\n) must be removed
                self.categories[line[:-1]] = i

        # Load the image paths
        self.data = []
        # For the moment I use only the first split
        file_name = os.path.join(root_dir, set_type + '.txt')
        with open(file_name, 'r') as f:
            for line in f:
                # Each row in self.data is composed by:
                # [label, img_path, crop_area]
                tmp = line.split(',')
                label = int(tmp[0])
                path = os.path.join(self.root_dir, 'photo_orig',
                                    tmp[1][-1], tmp[1] + '.jpg')
                # The last line char (\n) must be removed
                patch_center = [float(tmp[2]), float(tmp[3][:-1])]
                self.data.append([label, path, patch_center])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx][1]
        image = Image.open(img_path)
        # Sometimes the images are opened as grayscale, so I need to force RGB
        image = image.convert('RGB')
        patch_center = self.data[idx][2]
        patch_center = [patch_center[0] * image.width,
                        patch_center[1] * image.height]

        if image.width < image.height:
            patch_size = image.width * self.scale
        else:
            patch_size = image.height * self.scale
        box = (patch_center[0] - patch_size / 2,
               patch_center[1] - patch_size / 2,
               patch_center[0] + patch_size / 2,
               patch_center[1] + patch_size / 2)
        print(box)
        image = image.crop(box)
        if self.transform:
            image = self.transform(image)
            if is_tensor(image):
                image = transforms.Normalize(self.mean, self.std)(image)

        return image, self.data[idx][0]
