import random

import numpy as np
import torch
from torch.utils.data import DataLoader

# Custom
from config import *
from data.sampler import SubsetSequentialSampler
from methods.kcenterGreedy import kCenterGreedy


def BCEAdjLoss(scores, lbl, nlbl, l_adj):
    lnl = torch.log(scores[lbl])
    lnu = torch.log(1 - scores[nlbl])
    labeled_score = torch.mean(lnl)
    unlabeled_score = torch.mean(lnu)
    bce_adj_loss = -labeled_score - l_adj * unlabeled_score
    return bce_adj_loss


def aff_to_adj(x, y=None):
    x = x.detach().cpu().numpy()
    adj = np.matmul(x, x.transpose())
    adj += -1.0 * np.eye(adj.shape[0])
    adj_diag = np.sum(adj, axis=0)  # rowise sum
    adj = np.matmul(adj, np.diag(1 / adj_diag))
    adj = adj + np.eye(adj.shape[0])
    adj = torch.Tensor(adj).cuda()

    return adj


def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label, _ in dataloader:
                yield img, label
    else:
        while True:
            for img, _, _ in dataloader:
                yield img


def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        uncertainty = torch.tensor([])

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
            _, _, features = models['backbone'](inputs)
            pred_loss = models['module'](features)  # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))
            pred_loss = pred_loss.detach().cpu()
            uncertainty = torch.cat((uncertainty, pred_loss), 0)

    return uncertainty


def get_Top1(models, unlabeled_loader):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        uncertainty = torch.tensor([])

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()

            scores, _, _ = models['backbone'](inputs)
            prob = torch.softmax(scores, dim=1)

            top1 = torch.topk(scores, k=1, dim=1)[0]
            top1 = -top1.detach().cpu()
            uncertainty = torch.cat((uncertainty, top1), 0)

    return uncertainty.cpu()


def get_Top2(models, unlabeled_loader):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        uncertainty = torch.tensor([])

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()

            scores, _, _ = models['backbone'](inputs)
            prob = torch.softmax(scores, dim=1)

            top2 = torch.topk(scores, k=2, dim=1)[0]
            top2 = -(top2[:, 0] - top2[:, 1]).detach().cpu()
            uncertainty = torch.cat((uncertainty, top2), 0)

    return uncertainty.cpu()


def get_max_entropy(models, unlabeled_loader):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        uncertainty = torch.tensor([])

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()

            scores, _, _ = models['backbone'](inputs)
            prob = torch.softmax(scores, dim=1)

            entropy = -(prob * torch.log(prob)).sum(dim=1)
            entropy = entropy.detach().cpu()
            uncertainty = torch.cat((uncertainty, entropy), 0)

    return uncertainty.cpu()


def get_cumulative_entropy(models, unlabeled_loader, args=None):
    models['backbone'].eval()
    models['module'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        sub_logit_all = torch.tensor([])
        pred_label_all = torch.tensor([])

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
            main_logit, _, features = models['backbone'](inputs)

            _, pred_label = torch.max(main_logit, dim=1)
            pred_label = pred_label.detach().cpu()
            pred_label_all = torch.cat((pred_label_all, pred_label), 0)

            sub_logit = models['module'](features)
            sub_logit = sub_logit.detach().cpu()
            sub_logit_all = torch.cat((sub_logit_all, sub_logit), 0)

    sub_prob = torch.softmax(sub_logit_all, dim=1)

    # get uncertainty
    if args.query != "None":
        if args.query == 'AUM':
            n_classes = sub_prob.size(1)
            sub_assigned_prob_onehot = torch.eye(n_classes)[pred_label_all.type(torch.int64)] * sub_prob
            sub_assigned_prob = torch.sum(sub_assigned_prob_onehot, dim=1)
            sub_second_prob = torch.max(sub_prob - sub_assigned_prob_onehot, dim=1)[0]
            AUM = sub_assigned_prob - sub_second_prob
            uncertainty = -AUM
        elif args.query == 'Entropy':
            sub_entropy = -(sub_prob * torch.log(sub_prob)).sum(dim=1)  # Entropy
            uncertainty = sub_entropy

    return uncertainty.cpu()


# Derived from
# https://github.com/Javadzb/Class-Balanced-AL/blob/4f89a4b1c49a1b71178da4f5e2a6caa0db773443/query_strategies/bayesian_active_learning_disagreement_dropout.py#L5
def predict_prob_dropout_split(models, unlabeled_loader, n_drop=10):
    models['backbone'].eval()
    for m in models['backbone'].modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        probs = torch.tensor([])
        # targets = unlabeled_loader.dataset.targets
        # probs = torch.zeros([n_drop, len(targets), len(np.unique(targets))]).cuda()

        with torch.no_grad():
            for inputs, _, _ in unlabeled_loader:
                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    inputs = inputs.cuda()

                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    batch_probs = torch.tensor([])

                for i in range(n_drop):
                    scores, _, _ = models['backbone'](inputs, method='BALD')
                    prob = torch.softmax(scores, dim=1).unsqueeze(1)  # [B,        1, N_CLS]
                    prob = prob.detach().cpu()
                    batch_probs = torch.cat((batch_probs, prob), dim=1)  # [B,   N_DROP, N_CLS]

                probs = torch.cat((probs, batch_probs), dim=0)  # [B*K, N_DROP, N_CLS], where K is N_BATCH

        pb = probs.mean(1)
        entropy1 = (-pb * torch.log(pb)).sum(1)
        entropy2 = (-probs * torch.log(probs)).sum(2).mean(1)

    uncertainty = entropy1 - entropy2

    return uncertainty.cpu()


def get_features(models, unlabeled_loader):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([]).cuda()
    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
                _, features_batch, _ = models['backbone'](inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features  # .detach().cpu().numpy()
    return feat


def get_kcg(models, labeled_data_size, unlabeled_loader, args):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([])

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
            _, features_batch, _ = models['backbone'](inputs)
            features_batch = features_batch.detach().cpu()
            features = torch.cat((features, features_batch), 0)
        feat = features.numpy()

        num_subset = len(unlabeled_loader.sampler) - labeled_data_size
        new_av_idx = np.arange(num_subset, (num_subset + labeled_data_size))
        sampling = kCenterGreedy(feat)
        batch = sampling.select_batch_(new_av_idx, args.add_num)
        other_idx = [x for x in range(num_subset) if x not in batch]
    return other_idx + batch


# Select the indices of the unlablled data according to the methods
def query_samples(model, method, data_unlabeled, subset, labeled_set, cycle, args):
    if method == 'Random':
        arg = [i for i in range(len(subset))]
        random.shuffle(arg)

    if method == 'Entropy':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH * 2,
                                      sampler=SubsetSequentialSampler(subset),
                                      pin_memory=True, num_workers=args.num_workers)

        # Measure uncertainty of each data points in the subset
        uncertainty = get_max_entropy(model, unlabeled_loader)
        arg = np.argsort(uncertainty)

    if method == 'TiDAL':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH * 2,
                                      sampler=SubsetSequentialSampler(subset),
                                      pin_memory=True, num_workers=args.num_workers)

        uncertainty = get_cumulative_entropy(model, unlabeled_loader, args)
        arg = np.argsort(uncertainty)

    if method == 'BALD':
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH * 2,
                                      sampler=SubsetSequentialSampler(subset),
                                      pin_memory=True, num_workers=args.num_workers)
        # Measure uncertainty of each data points in the subset
        uncertainty = predict_prob_dropout_split(model, unlabeled_loader, n_drop=10)
        arg = np.argsort(uncertainty)

    if method == 'CoreSet':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH * 2,
                                      sampler=SubsetSequentialSampler(subset + labeled_set),
                                      # more convenient if we maintain the order of subset
                                      pin_memory=True, num_workers=args.num_workers)

        arg = get_kcg(model, args.add_num * (cycle + 1), unlabeled_loader, args)

    if method == 'lloss':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH * 2,
                                      sampler=SubsetSequentialSampler(subset),
                                      pin_memory=True, num_workers=args.num_workers)

        # Measure uncertainty of each data points in the subset
        uncertainty = get_uncertainty(model, unlabeled_loader)
        arg = np.argsort(uncertainty)

    return arg
