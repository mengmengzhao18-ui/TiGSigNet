import torch
import sklearn.metrics as sk_metrics
import scipy.special
import numpy as np
from torch.nn.functional import threshold


def accuracy(distances, y, step=0.0001):
    min_threshold_d = min(distances)
    max_threshold_d = max(distances)
    
    # if max_threshold_d !=min_threshold_d:
    #     step = max(step, (max_threshold_d-min_threshold_d)/10000) 
    max_acc = 0
    same_id = (y == 1)
    beat_threshold = 0

    pos_ave_dist = torch.mean(distances[y==1])
    pos_max_dist = torch.max(distances[y==1])
    pos_min_dist = torch.min(distances[y==1])

    neg_ave_dist = torch.mean(distances[y==0])
    neg_max_dist = torch.max(distances[y==0])
    neg_min_dist = torch.min(distances[y==0])

    print(f"pos_ave:{pos_ave_dist}, pos_max:{pos_max_dist}, pos_min:{pos_min_dist}")
    print(f"neg_ave:{neg_ave_dist}, neg_max:{neg_max_dist}, neg_min:{neg_min_dist}")

    thresholds = torch.linspace(min_threshold_d, max_threshold_d, steps=10000)
    for threshold in thresholds:
        true_positive = torch.sum((distances <= threshold) & same_id).item()
        # true_positive_rate = true_positive.sum().float() / same_id.sum().float()
        true_negative = torch.sum((distances > threshold) & (~same_id)).item()
        # true_negative_rate = true_negative.sum().float() / (~same_id).sum().float()

        acc = (true_positive + true_negative)/len(y)
        if acc >= max_acc:
            max_acc = acc
            beat_threshold = threshold

    return max_acc, beat_threshold

def calculate_auc(distances, y, threshold):
    true = distances[y==1]
    false = distances[y==0]
    num_pos = float(torch.sum(y))
    num_neg = len(y) - num_pos

    FRR = torch.sum(((true > threshold)/num_pos).to(torch.float32))
    FAR = torch.sum(((false <= threshold)/num_neg).to(torch.float32))

    y_scores = scipy.special.expit(-distances)
    AUC = sk_metrics.roc_auc_score(y, y_scores)

    fpr, tpr, thresholds = sk_metrics.roc_curve(y, y_scores)
    fnr = 1 - tpr
    # 寻找FPR和FNR最接近的点
    abs_diffs = np.abs(fpr - fnr)
    min_diff_index = np.argmin(abs_diffs)
    # eer_threshold = thresholds[min_diff_index]
    EER = (fpr[min_diff_index] + fnr[min_diff_index]) / 2  

    return FRR,FAR,AUC,EER
