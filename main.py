# Python
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
import os

# Custom
import models.resnet as resnet
from models.query_models import LossNet
from train_test.train_test import train, test
from data.load_dataset import load_dataset, sync_dataset
from methods.selection_methods import query_samples
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="cifar10",
                    help="cifar10 / cifar100")
parser.add_argument("-i", "--imb_factor", type=int, default=100,
                    help="1 / 10 / 100")
parser.add_argument("--init_dist", type=str, default='random',
                    help="uniform / random.")

parser.add_argument("-m", "--method_type", type=str, default="TiDAL",
                    help="")
parser.add_argument("-c", "--cycles", type=int, default=10,
                    help="Number of active learning cycles")
parser.add_argument("-t", "--total", type=bool, default=False,
                    help="Training on the entire dataset")
parser.add_argument("--seed", type=int, default=0,
                    help="Training seed.")
parser.add_argument("--subset", type=int, default=10000,
                    help="The size of subset.")
parser.add_argument("-q", "--query", type=str, default="Entropy",
                    help="The size of subset.")

args = parser.parse_args()

# Seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# balanced setting
if args.imb_factor == 1:
    args.add_num = {
        'cifar10': 1000,
        'cifar100': 2000,
        'fashionmnist': 1000,
        'svhn': 1000,
        'tiny-imagenet': 5000,
        'iNaturalist18': 20000,
    }[args.dataset]
else:
    args.add_num = {
        'cifar10': 500,
        'cifar100': 1000,
        'fashionmnist': 500,
        'svhn': 500,
        'tiny-imagenet': 5000,
    }[args.dataset]

args.subset = {
    'cifar10': 10000,
    'cifar100': 10000,
    'fashionmnist': 10000,
    'svhn': 10000,
    'tiny-imagenet': 20000,
    'iNaturalist18': 100000,
}[args.dataset]

if args.dataset == 'tiny-imagenet':
    args.initial_size = 10000
else:
    args.initial_size = args.add_num

args.num_workers = 4 if args.dataset == 'iNaturalist18' else 0

##
# Main
if __name__ == '__main__':

    method = args.method_type
    methods = ['Random', 'Entropy', 'BALD', 'CoreSet', 'lloss', 'TiDAL']
    datasets = ['cifar10', 'cifar100']
    assert method in methods, 'No method %s! Try options %s' % (method, methods)
    assert args.dataset in datasets, 'No dataset %s! Try options %s' % (args.dataset, datasets)
    '''
    method_type: 'Random', 'Entropy', 'CoreSet', 'lloss', 'TiDAL'
    '''
    os.makedirs('../results', exist_ok=True)
    txt_name = f'../results/results_{args.dataset}_{str(args.imb_factor)}_{str(args.method_type)}_{args.query}.txt'
    results = open(txt_name, 'w')

    print(txt_name)
    print("Dataset: %s" % args.dataset)
    print("Method type:%s" % method)
    if args.total:
        TRIALS = 1
        CYCLES = 1
    else:
        CYCLES = args.cycles

    for trial in range(TRIALS):
        # Load training and testing dataset
        data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train = load_dataset(args)
        print('The entire datasize is {}'.format(len(data_train)))
        NUM_TRAIN = no_train
        indices = list(range(NUM_TRAIN))

        if args.total:
            labeled_set = indices
        else:
            labeled_set = indices[:args.add_num]
            unlabeled_set = [x for x in indices if x not in labeled_set]

        train_loader = DataLoader(data_train, batch_size=BATCH,
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True, num_workers=args.num_workers)
        test_loader = DataLoader(data_test, batch_size=BATCH,
                                 pin_memory=True, num_workers=args.num_workers)
        dataloaders = {'train': train_loader, 'test': test_loader}

        # Model - create new instance for every trial so that it resets
        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            resnet_ = resnet.ResNet18(num_classes=NO_CLASSES)

            if (method == 'lloss') or (method == 'TiDAL'):
                # loss_module = LossNet(feature_sizes=[16,8,4,2], num_channels=[128,128,256,512]).cuda()
                out_dim = NO_CLASSES if (method == 'TiDAL') else 1
                pred_module = LossNet(out_dim=out_dim)

        models = {'backbone': resnet_}
        if (method == 'lloss') or (method == 'TiDAL'):
            models = {'backbone': resnet_, 'module': pred_module}

        # Loss, criterion and scheduler (re)initialization
        criterion = {}
        criterion['CE'] = nn.CrossEntropyLoss(reduction='none')
        criterion['KL_Div'] = nn.KLDivLoss(reduction='batchmean')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for key, val in models.items():
            models[key] = models[key].to(device)

        for cycle in range(CYCLES):
            # Randomly sample 10000 unlabeled data points
            if not args.total:
                random.shuffle(unlabeled_set)
                subset = unlabeled_set[:args.subset]

            torch.backends.cudnn.benchmark = True
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR,
                                       momentum=MOMENTUM, weight_decay=WDECAY)

            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            optimizers = {'backbone': optim_backbone}
            schedulers = {'backbone': sched_backbone}
            if (method == 'lloss') or (method == 'TiDAL'):
                optim_module = optim.SGD(models['module'].parameters(), lr=LR,
                                         momentum=MOMENTUM, weight_decay=WDECAY)
                sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
                optimizers = {'backbone': optim_backbone, 'module': optim_module}
                schedulers = {'backbone': sched_backbone, 'module': sched_module}

            # Training and testing
            train(models, method, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL)
            acc = test(models, EPOCH, method, dataloaders, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, TRIALS, cycle + 1,
                                                                                        CYCLES, len(labeled_set), acc))
            np.array([method, trial + 1, TRIALS, cycle + 1, CYCLES, len(labeled_set), acc]).tofile(results, sep=" ")
            results.write("\n")

            if cycle == (CYCLES - 1):
                # Reached final training cycle
                print("Finished.")
                break
            # Get the indices of the unlabeled samples to train on next cycle
            arg = query_samples(models, method, data_unlabeled, subset, labeled_set, cycle, args)

            # Update the labeled dataset and the unlabeled dataset, respectively
            new_list = list(torch.tensor(subset)[arg][:args.add_num].numpy())

            # print(len(new_list), min(new_list), max(new_list))
            labeled_set += list(torch.tensor(subset)[arg][-args.add_num:].numpy())
            listd = list(torch.tensor(subset)[arg][:-args.add_num].numpy())
            unlabeled_set = listd + unlabeled_set[args.subset:]
            print(len(labeled_set), min(labeled_set), max(labeled_set))

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True, num_workers=args.num_workers)

    results.close()
