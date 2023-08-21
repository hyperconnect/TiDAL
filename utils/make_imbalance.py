import numpy as np
import pandas as pd
import random

def get_img_num_per_cls_unif(dataset, cls_num=10, imb_type='exp', imb_factor=50, reverse=False):
    imb_ratio = 1 / imb_factor
    img_max = len(dataset.unlabeled_data) / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            if reverse:
                num = img_max * (imb_ratio ** ((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
            else:
                num = img_max * (imb_ratio ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_ratio))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls


def gen_imbalanced_data_unif(dataset, img_num_per_cls):
    new_data = []
    new_targets = []
    targets_np = np.array(dataset.unlabeled_targets, dtype=np.int64)
    classes = np.unique(targets_np)

    dataset.num_per_cls_dict = dict()
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        dataset.num_per_cls_dict[the_class] = the_img_num
        idx = np.where(targets_np == the_class)[0]
        np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]
        new_data.append(dataset.unlabeled_data[selec_idx, ...])
        new_targets.extend([the_class, ] * the_img_num)

    new_data = np.vstack(new_data)
    new_idx = [i for i in range(len(new_data))]
    random.shuffle(new_idx)
    dataset.data = np.vstack((dataset.data, new_data[new_idx]))
    dataset.targets = dataset.targets + list(np.array(new_targets)[new_idx])

    return dataset
