from __future__ import print_function
import argparse
from torchvision import models


class PrintNetList(argparse.Action):
    def __init__(self, option_strings, dest, nargs=0, **kwargs):
        super(PrintNetList, self).__init__(option_strings, dest, nargs,
                                           **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print("Network implementations from the Torchvision module:")
        head = "COMMAND               NETWORK"
        print(head)
        net_col = len(head) - len("NETWORK")
        cmds = ["tv-alexnet", "tv-vgg11", "tv-vgg11_bn", "tv-vgg13",
                "tv-vgg13_bn", "tv-vgg16", "tv-vgg16_bn", "tv-vgg19",
                "tv-vgg19_bn", "tv-resnet18", "tv-resnet34",
                "tv-resnet50", "tv-resnet101", "tv-resnet152",
                "tv-squeezenet1_0", "tv-squeezenet1_1", "tv-densenet121",
                "tv-densenet161", "tv-densenet169", "tv-densenet201",
                "tv-inception_v3"]
        nets = ["AlexNet", "VGG 11-layers", "VGG 11-layers with batch norm.",
                "VGG 13-layers", "VGG 13-layers with batch norm.",
                "VGG 16-layers", "VGG 16 layers with batch norm.",
                "VGG 19-layers", "VGG 19-layers with batch norm.",
                "ResNet-18", "ResNet-34", "ResNet-50", "ResNet-101",
                "ResNet-152", "SqueezeNet 1.0", "SqueezeNet 1.1",
                "DenseNet-121 (BC)", "DenseNet-161 (BC)",
                "DenseNet-169 (BC)", "DenseNet-201 (BC)",
                "Inception v3"]

        for cmd, net in zip(cmds, nets):
            print(cmd, end='')
            for char in range(net_col - len(cmd)):
                print(" ", end='')
            print(net)

        setattr(namespace, self.dest, True)


def torchvision_model(name, num_class):
    net_builder = getattr(models, name)
    if (name == "inception_v3"):
        net = net_builder(num_classes=num_class, aux_logits=False)
    else:
        net = net_builder(num_classes=num_class)

    return net


def get_model(model_name, num_class):
    net = None

    if model_name[:2] == "tv":
        # Torchvision model
        net = torchvision_model(model_name[3:], num_class)

    return net


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Model parsing module')

    parser.add_argument('--net-list', nargs=0, action=PrintNetList,
                        help='Print the list of the available network' +
                        'architectures')

    args = parser.parse_args()
