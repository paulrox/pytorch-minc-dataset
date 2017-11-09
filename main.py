from __future__ import print_function
import numpy as np
import pandas as pd
import argparse
import json
import os
import sys
import shortuuid
import platform
import ast
from time import strftime, time
import visdom
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model_parser import get_model, PrintNetList
from datasets.minc2500 import MINC2500
from datasets.minc import MINC
from cmstats import updateCM, MulticlassStat


def main():
    global net

    # Model and data parameters
    model = args.model
    dataset = args.dataset
    batch_size = args.batch_size
    classes = ast.literal_eval(args.classes)
    gpu = args.gpu
    seed = args.seed

    # Training parameters
    method = args.method
    epochs = args.epochs
    momentum = args.momentum
    w_decay = args.w_decay

    # Learning rate scheduler parameters
    l_rate = args.l_rate
    scheduler = args.lrate_sched
    step_size = args.step_size
    milestones = ast.literal_eval(args.milestones)
    gamma = args.gamma

    # Start training from scratch
    if not args.resume and not args.test:
        # Load the network model
        net = get_model(model, len(classes))
        if net is None:
            print("Unknown model name:", model + ".",
                  "Use '--net-list' option",
                  "to check the available network models")
            sys.exit(2)
        if gpu > 0:
            net.cuda()

        # Initialize the random generator
        torch.manual_seed(seed)
        if gpu > 0:
            torch.cuda.manual_seed_all(seed)

        # Dictionary used to store the training results and metadata
        json_data = {"platform": platform.platform(),
                     "date": strftime("%Y-%m-%d_%H:%M:%S"), "impl": "pytorch",
                     "dataset": dataset, "gpu": gpu,
                     "model": model, "epochs": epochs,
                     "classes": classes}
        json_data["train_params"] = {"method": method,
                                     "batch_size": batch_size,
                                     "seed": seed,
                                     "last_epoch": 0,
                                     "train_time": 0.0}
        epochs = range(epochs)

        # Optimization method
        if method == "SGD":
            optimizer = torch.optim.SGD(net.parameters(),
                                        lr=l_rate,
                                        momentum=momentum,
                                        weight_decay=w_decay)

        # Learning rate scheduler
        lrate_dict = dict()
        lrate_dict["sched"] = args.lrate_sched
        if args.lrate_sched is not "constant":
            if args.lrate_sched == "step":
                lrate_dict["step_size"] = step_size
                lrate_dict["gamma"] = gamma
                scheduler = lr_sched.StepLR(optimizer, step_size, gamma)
            elif args.lrate_sched == "multistep":
                lrate_dict["milestones"] = milestones
                lrate_dict["gamma"] = gamma
                scheduler = lr_sched.MultiStepLR(optimizer, milestones, gamma)
            elif args.lrate_sched == "exponential":
                lrate_dict["gamma"] = gamma
                scheduler = lr_sched.ExponentialLR(optimizer, gamma)
        json_data["train_params"]["l_rate"] = lrate_dict

        # Extract training parameters from the optimizer state
        for t_param in optimizer.state_dict()["param_groups"][0]:
            if t_param is not "params":
                json_data["train_params"][t_param] = \
                    optimizer.state_dict()["param_groups"][0][t_param]

        num_par = 0
        for parameter in net.parameters():
            num_par += parameter.numel()
        json_data["num_params"] = num_par

    # Resume from a training checkpoint or test the network
    else:
        with open(args.resume or args.test, 'rb') as f:
            json_data = json.load(f)
        train_info = json_data["train_params"]
        dataset = json_data["dataset"]
        batch_size = train_info["batch_size"]
        torch.manual_seed(train_info["seed"])

        if json_data["gpu"] > 0:
            torch.cuda.manual_seed_all(train_info["seed"])

        # Load the network model
        classes = json_data["classes"]
        net = get_model(json_data["model"], len(classes))
        if (json_data["gpu"] > 0):
            net.cuda()

        if args.resume:
            # Resume training
            # Load the saved state
            # (in the same directory as the json file)
            last_epoch = train_info["last_epoch"]
            epochs = range(last_epoch, json_data["epochs"])
            chk_dir = os.path.split(args.resume)[0]
            state = torch.load(os.path.join(chk_dir, json_data["state"]))

            # Load the network parameters
            net.load_state_dict(state["params"])

            # Load the optimizer state
            method = train_info["method"]
            if method == "SGD":
                optimizer = torch.optim.SGD(net.parameters(),
                                            lr=train_info["initial_lr"])
                optimizer.load_state_dict(state["optim"])

            # Load the learning rate scheduler info
            if train_info["l_rate"]["sched"] == "step":
                step_size = train_info["l_rate"]["step_size"]
                gamma = train_info["l_rate"]["gamma"]
                scheduler = lr_sched.StepLR(optimizer, step_size, gamma,
                                            last_epoch)
            elif train_info["l_rate"]["sched"] == "multistep":
                milestones = train_info["l_rate"]["milestones"]
                gamma = train_info["l_rate"]["gamma"]
                scheduler = lr_sched.MultiStepLR(optimizer, milestones, gamma,
                                                 last_epoch)
            elif args.lrate_sched == "exponential":
                gamma = train_info["l_rate"]["gamma"]
                scheduler = lr_sched.ExponentialLR(optimizer, gamma,
                                                   last_epoch)

        else:
            # Test the network
            # Load the saved parameters
            # (in the same directory as the json file)
            res_dir = os.path.split(args.test)[0]
            if "params" in json_data:
                net.load_state_dict(torch.load(os.path.join(res_dir,
                                                            json_data["params"]
                                                            )))
            elif "state" in json_data:
                # Test a checkpointed network
                state = torch.load(os.path.join(res_dir, json_data["state"]))
                net.load_state_dict(state["params"])
            else:
                sys.exit("No network parameters found in JSON file")

    if args.data_root:
        data_root = args.data_root
    else:
        # Default directory
        data_root = os.path.join(os.curdir, dataset + "_root")

    # Prepare data structures
    if not args.test:
        # Training phase
        train_trans = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        val_trans = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        if dataset == "minc2500":
            train_set = MINC2500(root_dir=data_root, set_type='train',
                                 split=1, transform=train_trans)
            val_set = MINC2500(root_dir=data_root, set_type='validate',
                               split=1, transform=val_trans)
        else:
            train_set = MINC(root_dir=data_root, set_type='train',
                             classes=classes, transform=train_trans)
            val_set = MINC(root_dir=data_root, set_type='validate',
                           classes=classes, transform=val_trans)

        train_loader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  shuffle=True, num_workers=args.workers,
                                  pin_memory=(args.gpu > 0))
        val_loader = DataLoader(dataset=val_set,
                                batch_size=batch_size,
                                shuffle=False, num_workers=args.workers,
                                pin_memory=(args.gpu > 0))

        # Loss function
        if gpu > 0:
            criterion = nn.CrossEntropyLoss().cuda()
        else:
            criterion = nn.CrossEntropyLoss()

        # Visdom windows to draw the training graphs
        loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                               Y=torch.zeros((1)).cpu(),
                               opts=dict(xlabel='Iteration (batch size = ' +
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
                                         title='Validation Precision (Macro)',
                                         legend=['Precision']))
        recall_window = vis.line(X=torch.zeros((1,)).cpu(),
                                 Y=torch.zeros((1)).cpu(),
                                 opts=dict(xlabel='Epoch',
                                           ylabel='Recall',
                                           title='Validation Recall (Macro)',
                                           legend=['Recall']))
        val_windows = [acc_window, prec_window, recall_window]

    # Testing phase
    test_trans = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    if dataset == "minc2500":
        test_set = MINC2500(root_dir=data_root, set_type='test', split=1,
                            transform=test_trans)
    else:
        test_set = MINC(root_dir=data_root, set_type='test',
                        classes=classes, transform=test_trans)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size,
                             shuffle=False, num_workers=args.workers,
                             pin_memory=(args.gpu > 0))

    if not args.test:

        # Training loop
        print("Training network on the", len(train_set), "training examples")
        for epoch in epochs:
            start_epoch = time()

            # Train the Model
            scheduler.step()
            train(train_loader, criterion, optimizer, epoch, epochs,
                  loss_window)

            # Check accuracy on validation set
            print("Validating network on the", len(val_set),
                  "validation images...")
            validate(val_loader, epoch, len(classes), val_windows)
            json_data["train_params"]["train_time"] += round(time() -
                                                             start_epoch, 3)

            # Save the checkpoint state
            save_state(net, optimizer, json_data, epoch + 1, args.chk_dir)

    # Test the model on the testing set
    print("Testing network on the", len(test_set), "testing images...")
    test(test_loader, args, json_data)
    # Save the trained network parameters and the testing results
    save_params(net, json_data, args.save_dir)


def train(train_loader, criterion, optimizer, epoch, epochs,
          loss_window):
    # Switch to train mode
    net.train()

    for i, (images, labels) in enumerate(train_loader):
        if args.gpu > 0:
            images = Variable(images.cuda(async=True))
            labels = Variable(labels.cuda(async=True))
        else:
            images = Variable(images)
            labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = net(images)
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


def validate(val_loader, epoch, n_class, val_windows):
    # Switch to evaluation mode
    net.eval()

    # Create the confusion matrix
    cm = np.zeros([n_class, n_class])
    for images, labels in val_loader:
        if args.gpu > 0:
            images = Variable(images.cuda(async=True), volatile=True)
        else:
            images = Variable(images)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        # Update the confusion matrix
        cm = updateCM(cm, predicted.cpu(), labels)

    stats = MulticlassStat(cm).get_stats_dict()
    acc = stats["accuracy"]
    prec = stats["precision_M"]
    Fscore = stats["Fscore_M"]
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
        Y=torch.ones((1, 1)).cpu() * Fscore,
        win=val_windows[2],
        update='append')
    print('Validation: accuracy of the model: %.2f %%'
          % (acc * 100))


def test(test_loader, args, json_data):
    # Switch to evaluation mode
    net.eval()

    test_time = 0.0
    scores = torch.Tensor()
    all_labels = torch.LongTensor()
    # Create the confusion matrix
    n_class = len(json_data["classes"])
    cm = np.zeros([n_class, n_class])
    for images, labels in test_loader:
        start_batch = time()
        if args.gpu > 0:
            images = Variable(images.cuda(async=True), volatile=True)
        else:
            images = Variable(images)

        outputs = net(images)
        scores = torch.cat((scores, outputs.cpu().data))
        all_labels = torch.cat((all_labels, labels))
        _, predicted = torch.max(outputs.data, 1)
        test_time += time() - start_batch
        # Update the confusion matrix
        cm = updateCM(cm, predicted.cpu(), labels)

    # Save the scores on the testing set
    f_name = os.path.join(args.save_dir, json_data["impl"] + "_" +
                          json_data["model"] + "_" +
                          json_data["dataset"] + "_" +
                          json_data["UUID"] + ".scores")
    torch.save(scores, f_name)

    # Compute the testing statistics and print them
    mc_stats = MulticlassStat(cm)
    print('******Test Results******')
    print('Time: ', round(test_time, 3), "seconds")
    mc_stats.print_stats()

    # Update the json data
    json_data["test_stats"] = mc_stats.get_stats_dict()
    json_data["test_stats"]["confusion_matrix"] = \
        pd.DataFrame(cm).to_dict(orient='split')
    json_data["test_stats"]["test_time"] = round(test_time, 6)

    # Plot the ROCs
    mc_stats.plot_multi_roc()
    mc_stats.plot_scores_roc(all_labels.numpy(), scores.numpy())


def save_state(net, optimizer, json_data, epoch, dir):
    json_data["train_params"]["last_epoch"] = epoch
    epoch_str = '_epoch_' + str(epoch)

    if epoch == 1:
        # Generate the UUID (8 characters long)
        id = shortuuid.uuid()[:8]
        json_data["UUID"] = id
    else:
        id = json_data["UUID"]

    f_name = os.path.join(dir, json_data["impl"] + "_" +
                          json_data["model"] + "_" +
                          json_data["dataset"] + "_" +
                          id + epoch_str)
    # Save training state
    state = dict()
    state["params"] = net.state_dict()
    state["optim"] = optimizer.state_dict()
    torch.save(state, f_name + '.state')

    # Update train parameters from optimizer state
    for t_param in state["optim"]["param_groups"][0]:
        if t_param is not "params":
            print(state["optim"])
            json_data["train_params"][t_param] = \
                state["optim"]["param_groups"][0][t_param]

    # Save experiment metadata
    json_data['state'] = os.path.split(f_name + '.state')[1]
    with open(f_name + ".json", 'wb') as f:
        json.dump(json_data, f)


def save_params(net, json_data, dir):
    if "last_epoch" in json_data["train_params"]:
        del json_data["train_params"]["last_epoch"]
    if "state" in json_data:
        del json_data["state"]

    f_name = os.path.join(dir, json_data["impl"] + "_" +
                          json_data["model"] + "_" +
                          json_data["dataset"] + "_" +
                          json_data["UUID"])
    # Save training state
    torch.save(net.state_dict(), f_name + '.state')
    # Save experiment metadata
    json_data['params'] = os.path.split(f_name + '.params')[1]
    with open(f_name + ".json", 'wb') as f:
        json.dump(json_data, f)


if __name__ == '__main__':

    vis = visdom.Visdom()

    parser = argparse.ArgumentParser(description='Train and test a network ' +
                                                 'on the MINC datasets')
    # Data Options
    data_args = parser.add_argument_group('Data arguments')
    data_args.add_argument('--dataset', metavar='NAME', default='minc2500',
                           choices=['minc2500', 'minc'],
                           help='name of the dataset to be used' +
                           ' (default: minc2500)')
    data_args.add_argument('--data-root', metavar='DIR', help='path to ' +
                           'dataset (default: ./$(DATASET)_root)')
    data_args.add_argument('--save-dir', metavar='DIR', default='./results',
                           help='path to trained models (default: results/)')
    data_args.add_argument('--chk-dir', metavar='DIR', default='./checkpoints',
                           help='path to checkpoints (default: checkpoints/)')
    data_args.add_argument('--workers', metavar='NUM', type=int,
                           default=8, help='number of worker threads for' +
                           ' the data loader')

    # Model Options
    model_args = parser.add_argument_group('Model arguments')
    model_args.add_argument('-m', '--model', metavar='NAME',
                            default='tv-densenet121', type=str,
                            help='name of the netwrok model to be used')
    model_args.add_argument('--classes', metavar='LIST',
                            default='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,' +
                            '16,17,18,19,20,21,22]',
                            help='indicies of the classes to be used for the' +
                            ' classification')

    # Training Options
    train_args = parser.add_argument_group('Training arguments')
    train_args.add_argument('--method', default='SGD', metavar='NAME',
                            help='training method to be used')
    train_args.add_argument('--gpu', type=int, default=1, metavar='NUM',
                            help='number of GPUs to use')
    train_args.add_argument('--epochs', default=20, type=int, metavar='NUM',
                            help='number of total epochs to run (default: 20)')
    train_args.add_argument('-b', '--batch-size', default=64, type=int,
                            metavar='NUM',
                            help='mini-batch size (default: 64)')
    train_args.add_argument('--momentum', type=float, default=0.9,
                            metavar='NUM', help='Momentum (default: 0.9)')
    train_args.add_argument('--w-decay', type=float, default=1e-4,
                            metavar='NUM', help='weigth decay (default: 1e-4)')
    train_args.add_argument('--seed', type=int, metavar='NUM',
                            default=179424691,
                            help='random seed (default: 179424691)')

    # Learning Rate Scheduler Options
    lrate_args = parser.add_argument_group('Learning rate arguments')
    lrate_args.add_argument('--l-rate', type=float, default=0.1,
                            metavar='NUM', help='initial learning Rate' +
                            ' (default: 0.1)')
    lrate_args.add_argument('--lrate-sched', default="multistep",
                            metavar="NAME", help="name of the learning " +
                            "rate scheduler (default: constant)",
                            choices=['step', 'multistep', 'exponential',
                                     'constant'])
    lrate_args.add_argument('--milestones', default='[5,10]', metavar='LIST',
                            help='epoch indicies for learning rate reduction' +
                            ' (multistep, default: [5,10])')
    lrate_args.add_argument('--gamma', type=float, default=0.1,
                            metavar='NUM', help='multiplicative factor of ' +
                            'learning rate decay (default: 0.1)')
    lrate_args.add_argument('--step-size', type=int, default=5,
                            metavar='NUM', help='pediod of learning rate ' +
                            'decay (step, default: 5)')

    # Other Options
    parser.add_argument('--resume', default='', type=str, metavar='JSON_FILE',
                        help='resume the training from the specified JSON ' +
                        'file  (default: none)')
    parser.add_argument('--test', default='', type=str, metavar='JSON_FILE',
                        help='test the network from the specified JSON file')
    parser.add_argument('--net-list', action=PrintNetList,
                        help='Print the list of the available network ' +
                        'architectures')

    args = parser.parse_args()

    if not args.net_list:
        main()
