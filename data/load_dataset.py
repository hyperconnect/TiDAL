import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100, CIFAR10

from config import *
from utils.make_imbalance import *


class MyDataset(Dataset):
    def __init__(self, dataset_name, train_flag=None, transf=None, args=None):
        self.dataset_name = dataset_name
        self.args = args

        if args is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True

        if self.dataset_name == "cifar10":
            self.dataset = CIFAR10('../cifar10', train=train_flag,
                                   download=True, transform=transf)
        elif self.dataset_name == "cifar100":
            self.dataset = CIFAR100('../cifar100', train=train_flag,
                                    download=True, transform=transf)

        if args is not None:
            targets = np.array(self.dataset.targets)
            classes, class_counts = np.unique(targets, return_counts=True)
            nb_classes = len(classes)
            self.moving_prob = np.zeros((len(self.dataset), nb_classes), dtype=np.float32)

            if args.init_dist == 'uniform':
                init_num_per_cls = int(args.initial_size / nb_classes)
            elif args.init_dist == 'random':
                init_num_per_cls = 1

            initial_idx = []
            for j in range(nb_classes):
                jth_cls_idx = [i for i, label in enumerate(targets) if label == j]
                random.shuffle(jth_cls_idx)
                initial_idx += jth_cls_idx[:init_num_per_cls]

            dataset_idx = [i for i in range(len(targets))]
            unlabel_idx = [item for item in dataset_idx if item not in initial_idx]

            temp_data = self.dataset.data
            temp_targets = self.dataset.targets

            self.dataset.data = temp_data[initial_idx]
            self.dataset.targets = list(np.array(temp_targets)[initial_idx])
            self.dataset.unlabeled_data = temp_data[unlabel_idx]
            self.dataset.unlabeled_targets = list(np.array(temp_targets)[unlabel_idx])

            self.img_num_per_cls = get_img_num_per_cls_unif(self.dataset, cls_num=nb_classes,
                                                            imb_factor=args.imb_factor)
            self.dataset = gen_imbalanced_data_unif(self.dataset, self.img_num_per_cls)

    def __getitem__(self, index):
        if self.args is not None:
            data, target = self.dataset[index]
            moving_prob = self.moving_prob[index]
            return data, target, index, moving_prob
        else:
            data, target = self.dataset[index]
            return data, target, index

    def __len__(self):
        return len(self.dataset)


def sync_dataset(labeled_set, unlabeled_set):
    unlabeled_set.dataset.data = labeled_set.dataset.data
    unlabeled_set.dataset.targets = labeled_set.dataset.targets
    return unlabeled_set


# Data
def load_dataset(args):
    dataset = args.dataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=4),
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
    ])

    if dataset == 'cifar10':
        data_train = MyDataset(dataset, train_flag=True, transf=train_transform, args=args)
        data_unlabeled = MyDataset(dataset, train_flag=True, transf=test_transform)
        data_unlabeled = sync_dataset(data_train, data_unlabeled)
        data_test = CIFAR10('../cifar10', train=False, download=True, transform=test_transform)
        data_test.pred_main = np.zeros((len(data_test), EPOCH, 10), dtype=np.float32)
        data_test.pred_sub = np.zeros((len(data_test), EPOCH, 10), dtype=np.float32)
        NO_CLASSES = 10
        NUM_TRAIN = len(data_train)
        no_train = NUM_TRAIN

    elif dataset == 'cifar100':
        data_train = MyDataset(dataset, train_flag=True, transf=train_transform, args=args)
        data_unlabeled = MyDataset(dataset, train_flag=True, transf=test_transform)
        data_unlabeled = sync_dataset(data_train, data_unlabeled)
        data_test = CIFAR100('../cifar100', train=False, download=True, transform=test_transform)
        NO_CLASSES = 100
        NUM_TRAIN = len(data_train)
        no_train = NUM_TRAIN

    adden = args.add_num
    return data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train
