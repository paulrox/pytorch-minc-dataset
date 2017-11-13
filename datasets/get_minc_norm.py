from __future__ import print_function
import time
import torch
from minc import MINC
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])

batch_size = 1

classes = [6, 9, 10, 11, 14, 15]

all_data = MINC(root_dir='../minc_root', set_type='all',
                classes=classes, transform=transform)
all_loader = torch.utils.data.DataLoader(dataset=all_data,
                                         batch_size=batch_size,
                                         num_workers=8)


def get_std(mean):
    global all_data, all_loader
    sum = torch.zeros(3)
    pixels = 0
    alpha = 0.9
    next_perc = 1
    est_time_mean = 0.0
    perc_time = time.time()

    for i, (imgs, labels) in enumerate(all_loader):
        sum[0] = sum[0] + torch.sum(torch.pow(torch.add(imgs[:, 0, :, :],
                                                        -mean[0]), 2))
        sum[1] = sum[1] + torch.sum(torch.pow(torch.add(imgs[:, 1, :, :],
                                                        -mean[1]), 2))
        sum[2] = sum[2] + torch.sum(torch.pow(torch.add(imgs[:, 2, :, :],
                                                        -mean[2]), 2))
        pixels += imgs.shape[2] + imgs.shape[3]

        perc = float(i * batch_size * 100) / float(len(all_data))
        if int(perc * 10) == next_perc:
            est_time_ist = time.time() - perc_time
            if int(perc * 10) == 1:
                est_time_mean = est_time_ist
            else:
                est_time_mean = (alpha * est_time_mean + (1 - alpha)
                                 * est_time_ist)

            remaining_mean = round(((100 - perc) * 10 * est_time_mean) / 60, 2)
            print("Std computation:", round(perc, 1),
                  "%  remaining time :", remaining_mean)
            next_perc += 1
            perc_time = time.time()

    print("SUM:", sum)
    print("Pixels:", pixels)

    sum[0] = sum[0] / pixels
    sum[1] = sum[1] / pixels
    sum[2] = sum[2] / pixels

    std = torch.sqrt(sum)

    print('Std: %f , %f , %f' % (std[0], std[1], std[2]))

    return std


def main():
    global all_data, all_loader

    print(all_data.categories)
    print("Total patch num: ", len(all_data))

    # Per-channel mean obtained from the MINC paper
    mean = torch.Tensor([0.48627451, 0.458823529, 0.407843137])

    # Compute the dataset std with the obtained mean values
    get_std(mean)

    print('Done')


if __name__ == '__main__':
    main()
