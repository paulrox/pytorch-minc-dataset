from __future__ import print_function
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MINC(Dataset):
    def __init__(self, root_dir, set_type='train', classes=range(23),
                 scale=0.233, transform=None):
        self.root_dir = root_dir
        self.set_type = set_type
        self.transform = transform
        self.scale = scale
        # This value has been obtained from the MINC paper
        self.mean = torch.Tensor([0.48627451, 0.458823529, 0.407843137])
        # self.std = Tensor([0.254254, 0.252448, 0.266003])

        # Get the material categories from the categories.txt file
        file_name = os.path.join(root_dir, 'categories.txt')
        self.categories = dict()
        new_class_id = 0
        with open(file_name, 'r') as f:
            for class_id, class_name in enumerate(f):
                if class_id in classes:
                    # The last line char (\n) must be removed
                    self.categories[class_id] = [class_name[:-1], new_class_id]
                    new_class_id += 1

        # Load the image data
        set_types = ['train', 'validate', 'test']
        self.data = []
        if set_type == "train":
            set_num = range(1)
        if set_type == "validate":
            set_num = range(1, 2)
        if set_type == "test":
            set_num = range(2, 3)
        if set_type == "all":
            set_num = range(3)
        for i in set_num:
            file_name = os.path.join(root_dir, set_types[i] + '.txt')
            with open(file_name, 'r') as f:
                for line in f:
                    # Each row in self.data is composed by:
                    # [label, img_path, patch_center]
                    tmp = line.split(',')
                    label = int(tmp[0])
                    # Check if the patch label is in the new class set
                    if label in self.categories:
                        img_id = tmp[1]
                        patch_x = float(tmp[2])
                        # The last line char (\n) must be removed
                        patch_y = float(tmp[3][:-1])
                        path = os.path.join(self.root_dir, 'photo_orig',
                                            img_id[-1], img_id + '.jpg')
                        patch_center = [patch_x, patch_y]
                        self.data.append([self.categories[label][1], path,
                                          patch_center])

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
            patch_size = int(image.width * self.scale)
        else:
            patch_size = int(image.height * self.scale)
        box = (patch_center[0] - patch_size / 2,
               patch_center[1] - patch_size / 2,
               patch_center[0] + patch_size / 2,
               patch_center[1] + patch_size / 2)
        image = image.crop(box)
        if self.transform:
            image = self.transform(image)
            if torch.is_tensor(image):
                torch.add(image[0, :, :], -self.mean[0])
                torch.add(image[1, :, :], -self.mean[1])
                torch.add(image[2, :, :], -self.mean[2])
                # image = transforms.Normalize(self.mean, self.std)(image)

        return image, self.data[idx][0]
