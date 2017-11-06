from __future__ import print_function
import numpy as np
import argparse
import json
import os
import sys
import shortuuid
import platform
from time import strftime, time
import visdom
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model_parser import get_model, PrintNetList
from minc2500 import MINC2500
from multiclassCM import confusionMatrix, getAccuracy, getPrecision, getRecall


def main():
    # Those values are computed using the script 'get_minc2500_norm.py'
    mean = torch.Tensor([0.507207, 0.458292, 0.404162])
    std = torch.Tensor([0.254254, 0.252448, 0.266003])

    net = get_model(args.model, args.n_class)
    if net is None:
        print("Unknown model name:", args.model + ".",
              "Use 'python model_parser.py --net-list'",
              "to check the available network models")
        sys.exit(2)
    if args.gpu > 0:
        net.cuda()

    batch_size = args.batch_size

    if not args.resume and not args.test:
        # Start training from scratch
        # Initialize the random generator
        if args.seed:
            seed = args.seed
            torch.manual_seed(seed)
            if args.gpu > 0:
                torch.cuda.manual_seed_all(seed)
        else:
            seed = torch.initial_seed()
            if args.gpu > 0:
                torch.cuda.manual_seed_all(seed)

        # Dictionary used to store the training results and metadata
        json_data = {"platform": platform.platform(),
                     "date": strftime("%Y-%m-%d_%H:%M:%S"), "impl": "pytorch",
                     "gpu": args.gpu, "dataset": "MINC-2500",
                     "model": args.model, "epochs": args.epochs}
        json_data["train"] = {"method": "SGD", "batch_size": args.batch_size,
                              "init_l_rate": args.l_rate,
                              "momentum": args.momentum,
                              "w_decay": args.w_decay,
                              "seed": seed,
                              "last_epoch": 0,
                              "train_time": 0.0}
        par = 0
        for parameter in net.parameters():
            par += parameter.numel()
        # The number of parameters is stored in milions
        par = par / 1e6
        json_data["num_params"] = par

        # Training parameters
        epochs = range(args.epochs)
        l_rate = args.l_rate
        momentum = args.momentum
        w_decay = args.w_decay

    elif args.resume:
        # Resume a training checkpoint
        with open(args.resume, 'rb') as f:
            json_data = json.load(f)
        train_info = json_data["train"]
        if train_info["method"] == "SGD":
            optimizer = torch.optim.SGD(net.parameters(),
                                        train_info["init_l_rate"],
                                        momentum=train_info["momentum"],
                                        weight_decay=train_info["w_decay"])
        epochs = range(train_info["last_epoch"], json_data["epochs"])
        batch_size = train_info["batch_size"]
        l_rate = train_info["init_l_rate"]
        momentum = train_info["momentum"]
        w_decay = train_info["w_decay"]
        torch.manual_seed(train_info["seed"])
        if json_data["gpu"] > 0:
            torch.cuda.manual_seed_all(train_info["seed"])
        # Load the saved parameters (in the same directory as the json file)
        chk_dir = os.path.split(args.resume)[0]
        net.load_state_dict(torch.load(os.path.join(chk_dir,
                                                    json_data["net_state"])))
    else:
        # Load the json data
        with open(args.test, 'rb') as f:
            json_data = json.load(f)
        seed = json_data["train"]["seed"]
        torch.manual_seed(seed)
        if json_data["gpu"] > 0:
            torch.cuda.manual_seed_all(long(seed))
        # Load the saved parameters (in the same directory as the json file)
        net_dir = os.path.split(args.test)[0]
        net.load_state_dict(torch.load(os.path.join(net_dir,
                                                    json_data["net_state"])))

    # Prepare data structures for the training phase
    if not args.test:
        train_trans = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        val_trans = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_set = MINC2500(root_dir=args.data_root, set_type='train',
                             split=1, transform=train_trans)
        val_set = MINC2500(root_dir=args.data_root, set_type='validate',
                           split=1, transform=val_trans)

        train_loader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  shuffle=True, num_workers=8,
                                  pin_memory=(args.gpu > 0))
        val_loader = DataLoader(dataset=val_set,
                                batch_size=batch_size,
                                shuffle=False, num_workers=8,
                                pin_memory=(args.gpu > 0))

        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(net.parameters(), l_rate,
                                    momentum=momentum,
                                    weight_decay=w_decay)
        # Visdom windows to draw the training graphs
        loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                               Y=torch.zeros((1)).cpu(),
                               opts=dict(xlabel='Minibatch (batch size = ' +
                                                str(batch_size) + ')',
                                         ylabel='Loss',
                                         title='Training Loss',
                                         legend=['Loss']))
        acc_window = vis.line(X=torch.zeros((1,)).cpu(),
                              Y=torch.zeros((1)).cpu(),
                              opts=dict(xlabel='Epoch',
                                        ylabel='Accuracy',
                                        title='Validation Accuracy',
                                        legend=['Accuracy']))
        prec_window = vis.line(X=torch.zeros((1,)).cpu(),
                               Y=torch.zeros((1)).cpu(),
                               opts=dict(xlabel='Epoch',
                                         ylabel='Precision',
                                         title='Validation Precision',
                                         legend=['Precision']))
        recall_window = vis.line(X=torch.zeros((1,)).cpu(),
                                 Y=torch.zeros((1)).cpu(),
                                 opts=dict(xlabel='Epoch',
                                           ylabel='Recall',
                                           title='Validation Recall',
                                           legend=['Recall']))
        val_windows = [acc_window, prec_window, recall_window]

    # Load the testing set
    test_trans = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_set = MINC2500(root_dir=args.data_root, set_type='test', split=1,
                        transform=test_trans)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size,
                             shuffle=False, num_workers=8,
                             pin_memory=(args.gpu > 0))

    if not args.test:
        for epoch in epochs:
            # Train the Model
            start_epoch = time()
            train(train_loader, net, criterion, optimizer, epoch, epochs,
                  loss_window)

            # Check accuracy on validation set
            validate(val_loader, net, epoch, args.n_class, val_windows)
            json_data["train"]["train_time"] += round(time() - start_epoch, 3)

            # Save the checkpoint state
            save_state(net, json_data, epoch + 1, args)

    # Test the model on the testing set
    test(test_loader, net, args.n_class, json_data)
    # Save the trained and tested model
    save_state(net, json_data, -1, args)


def train(train_loader, model, criterion, optimizer, epoch, epochs,
          loss_window):
    # Switch to train mode
    model.train()

    for i, (images, labels) in enumerate(train_loader):
        if args.gpu > 0:
            images = Variable(images.cuda(async=True))
            labels = Variable(labels.cuda(async=True))
        else:
            images = Variable(images)
            labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            vis.line(
                X=torch.ones((1, 1)).cpu() * ((epoch) * len(train_loader) + i),
                Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                win=loss_window,
                update='append')
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, epochs[-1] + 1, i, len(train_loader),
                     loss.data[0]))


def validate(val_loader, model, epoch, n_class, val_windows):
    # Switch to evaluation mode
    model.eval()

    # Create an empty confusion matrix
    cm = np.zeros([n_class, n_class])

    for images, labels in val_loader:
        if args.gpu > 0:
            images = Variable(images.cuda(async=True), volatile=True)
        else:
            images = Variable(images)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        # Update the confusion matrix
        cm = confusionMatrix(cm, predicted.cpu(), labels)

    acc = getAccuracy(cm)
    prec = getPrecision(cm)
    recall = getRecall(cm)
    vis.line(
        X=torch.ones((1, 1)).cpu() * (epoch + 1),
        Y=torch.ones((1, 1)).cpu() * acc,
        win=val_windows[0],
        update='append')
    vis.line(
        X=torch.ones((1, 1)).cpu() * (epoch + 1),
        Y=torch.ones((1, 1)).cpu() * prec,
        win=val_windows[1],
        update='append')
    vis.line(
        X=torch.ones((1, 1)).cpu() * (epoch + 1),
        Y=torch.ones((1, 1)).cpu() * recall,
        win=val_windows[2],
        update='append')
    print('Validation: accuracy of the model: %.2f %%'
          % (acc * 100))


def test(test_loader, model, n_class, json_data):
    # Switch to evaluation mode
    model.eval()

    # Create an empty confusion matrix
    cm = np.zeros([n_class, n_class])
    test_time = 0.0
    for images, labels in test_loader:
        start_batch = time()
        if args.gpu > 0:
            images = Variable(images.cuda(async=True), volatile=True)
        else:
            images = Variable(images)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_time += time() - start_batch
        # Update the confusion matrix
        cm = confusionMatrix(cm, predicted.cpu(), labels)

    print('******Test Results******')
    print('Time: ', round(test_time, 3), "seconds")
    acc = getAccuracy(cm)
    prec = getPrecision(cm)
    recall = getRecall(cm)
    print('Accuracy: %.2f %%'
          % (acc * 100))
    print('Precision: %.2f %%'
          % (prec * 100))
    print('Recall: %.2f %%'
          % (recall * 100))

    # Update the json data
    json_data["test_accuracy"] = round(acc, 4)
    json_data["test_precision"] = round(prec, 4)
    json_data["test_time"] = round(test_time, 6)


def save_state(net, json_data, epoch, args):
    if epoch == -1:
        dir = args.save_dir
        epoch_str = 'trained'
        del json_data["train"]["last_epoch"]
    else:
        json_data["train"]["last_epoch"] = epoch
        dir = args.chk_dir
        epoch_str = 'epoch_' + str(epoch)

    if epoch == 1:
        # Generate the UUID (8 characters long)
        id = shortuuid.uuid()[:8]
        json_data["UUID"] = id
    else:
        id = json_data["UUID"]

    f_name = os.path.join(dir, json_data["impl"] + "_" +
                          json_data["model"] + "_" +
                          json_data["dataset"] + "_" +
                          id + "_" + epoch_str)
    # Save model parameters
    torch.save(net.state_dict(), f_name + '.params')
    # Save experiment metadata
    json_data['net_state'] = os.path.split(f_name + '.params')[1]
    with open(f_name + ".json", 'wb') as f:
        json.dump(json_data, f)


if __name__ == '__main__':

    vis = visdom.Visdom()

    parser = argparse.ArgumentParser(description='Train and test a network' +
                                                 'on the MINC-2500 dataset')
    # Data Options
    parser.add_argument('--data-root', metavar='DIR', default='/media/paolo/' +
                        'Data/pytorch_datasets/minc-2500', help='path to ' +
                        'dataset (default: /media/paolo/Data/' +
                        'pytorch_datasets/minc-2500)')
    parser.add_argument('--save-dir', metavar='DIR', default='./trained',
                        help='path to trained models default(trained)')
    parser.add_argument('--chk-dir', metavar='DIR', default='./checkpoint',
                        help='path to checkpoints default(checkpoints)')
    parser.add_argument('--workers', metavar='W', type=int, default=8,
                        help='number of worker threads for the data loader')
    # Model Options
    parser.add_argument('--model', '-m', metavar='MODEL',
                        default='tv-densenet121', type=str)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--n-class', default=23, type=int, metavar='N',
                        help='number of classes to classify (default: 23)')
    parser.add_argument('--test', default='', type=str, metavar='PATH',
                        help='path to the parameters of the model' +
                             ' to be tested')
    # Training Options
    parser.add_argument('--gpu', type=int, default=1, metavar='G',
                        help='number of GPUs to use')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run (default: 20)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='b', help='mini-batch size (default: 64)')
    parser.add_argument('--l-rate', type=float, default=0.1, metavar='l',
                        help='Learning Rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='m',
                        help='Momentum (default: 0.9)')
    parser.add_argument('--w-decay', type=float, default=1e-4, metavar='wd',
                        help='weigth decay (default: 1e-4)')
    parser.add_argument('--seed', type=int, metavar='s',
                        default=179424691,
                        help='random seed (default: random)')
    # Other Options
    parser.add_argument('--net-list', action=PrintNetList,
                        help='Print the list of the available network' +
                        'architectures')

    args = parser.parse_args()

    if not args.net_list:
        main()
