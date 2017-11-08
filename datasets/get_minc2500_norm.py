from __future__ import print_function
import torch
from minc2500 import MINC2500
from torchvision import transforms

all_data = MINC2500(root_dir='../minc-2500_root',
                    set_type='all', transform=transforms.ToTensor())
all_loader = torch.utils.data.DataLoader(dataset=all_data,
                                         batch_size=100,
                                         num_workers=8)


def get_mean():
    global all_data, all_loader
    sum = torch.zeros(3)

    for i, (imgs, labels) in enumerate(all_loader):
        sum[0] = sum[0] + torch.sum(imgs[:, 0, :, :])
        sum[1] = sum[1] + torch.sum(imgs[:, 1, :, :])
        sum[2] = sum[2] + torch.sum(imgs[:, 2, :, :])

    mean = torch.Tensor([sum[0] / (len(all_data) * 362 * 362),
                         sum[1] / (len(all_data) * 362 * 362),
                         sum[2] / (len(all_data) * 362 * 362)])

    print('Mean: %f , %f , %f' % (mean[0], mean[1], mean[2]))

    return mean


def get_std(mean):
    global all_data, all_loader
    sum = torch.zeros(3)

    for i, (imgs, labels) in enumerate(all_loader):
        sum[0] = sum[0] + torch.sum(torch.pow(torch.add(imgs[:, 0, :, :],
                                                        -mean[0]), 2))
        sum[1] = sum[1] + torch.sum(torch.pow(torch.add(imgs[:, 1, :, :],
                                                        -mean[1]), 2))
        sum[2] = sum[2] + torch.sum(torch.pow(torch.add(imgs[:, 2, :, :],
                                                        -mean[2]), 2))

    sum[0] = sum[0] / (len(all_data) * 362 * 362 - 1)
    sum[1] = sum[1] / (len(all_data) * 362 * 362 - 1)
    sum[2] = sum[2] / (len(all_data) * 362 * 362 - 1)

    std = torch.sqrt(sum)

    print('Std: %f , %f , %f' % (std[0], std[1], std[2]))

    return std


def main():

    # Compute the dataset mean values for each color channel
    mean = get_mean()

    # Compute the dataset std with the obtained mean values
    get_std(mean)

    print('Done')


if __name__ == '__main__':
    main()
