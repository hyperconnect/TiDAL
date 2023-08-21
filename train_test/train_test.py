import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import *


def kl_div(source, target, reduction='batchmean'):
    loss = F.kl_div(F.log_softmax(source, 1), target, reduction=reduction)
    return loss


# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    criterion = nn.BCELoss()
    input = (input - input.flip(0))[:len(input) // 2]
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    diff = torch.sigmoid(input)
    one = torch.sign(torch.clamp(target, min=0))  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = criterion(diff, one)
    elif reduction == 'none':
        loss = criterion(diff, one)
    else:
        NotImplementedError()

    return loss


def test(models, method, dataloaders, mode='val'):
    assert mode in ['val', 'test']
    models['backbone'].eval()

    if method in ['TiDAL', 'lloss']:
        models['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
                labels = labels.cuda()

            scores, _, _ = models['backbone'](inputs)

            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total


def train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss):
    models['backbone'].train()
    if method in ['TiDAL', 'lloss']:
        models['module'].train()

    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            inputs = data[0].cuda()
            labels = data[1].cuda()
            index = data[2].detach().numpy().tolist()

        optimizers['backbone'].zero_grad()
        if method in ['TiDAL', 'lloss']:
            optimizers['module'].zero_grad()

        scores, emb, features = models['backbone'](inputs)
        target_loss = criterion['CE'](scores, labels)
        probs = torch.softmax(scores, dim=1)

        if method == 'TiDAL':
            moving_prob = data[3].cuda()
            moving_prob = (moving_prob * epoch + probs * 1) / (epoch + 1)
            dataloaders['train'].dataset.moving_prob[index, :] = moving_prob.cpu().detach().numpy()

            cumulative_logit = models['module'](features)
            m_module_loss = criterion['KL_Div'](F.log_softmax(cumulative_logit, 1), moving_prob.detach())
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            loss = m_backbone_loss + WEIGHT * m_module_loss

        elif method == 'lloss':
            if epoch > epoch_loss:
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()

            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))
            m_module_loss = LossPredLoss(pred_loss, target_loss, margin=MARGIN)

            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            loss = m_backbone_loss + WEIGHT * m_module_loss

        else:
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            loss = m_backbone_loss

        loss.backward()
        optimizers['backbone'].step()
        if method in ['TiDAL', 'lloss']:
            optimizers['module'].step()
    return loss


def train(models, method, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss):
    print('>> Train a Model.')
    best_acc = 0.

    for epoch in range(num_epochs):

        best_loss = torch.tensor([0.5]).cuda()
        loss = train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss)

        schedulers['backbone'].step()
        if method in ['TiDAL', 'lloss']:
            schedulers['module'].step()

        if False and epoch % 20 == 7:
            acc = test(models, method, dataloaders, mode='test')
            if best_acc < acc:
                best_acc = acc
                print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
    print('>> Finished.')
