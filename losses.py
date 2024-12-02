import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


class PseLoss(nn.Module):
    """Focal Loss"""

    def __init__(self, args):
        super(PseLoss, self).__init__()
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.scale = args.scale_factor
        self.threshold = args.threshold

    def forward(self, logits, masks, pseudo_label):
        bs = logits[0].shape[0] // 2
        logits = [x[bs:] for x in logits]
        masks = [x[bs:] for x in masks]
        pse_labels = [torch.max_pool1d(pseudo_label[bs:], kernel_size=self.scale**i) for i in range(len(logits))]
        pse_labels = [(x > self.threshold) for x in pse_labels]
        valid_mask = torch.cat(masks, dim=1)
        inputs = torch.cat(logits, dim=1)[valid_mask]
        targets = torch.cat(pse_labels, dim=1)[valid_mask]
        loss = sigmoid_focal_loss(inputs.float(), targets.float(), alpha=self.alpha, gamma=self.gamma, reduction="mean")
        return loss


class ClsLoss(nn.Module):
    def __init__(self, args):
        super(ClsLoss, self).__init__()
        self.criterion = nn.BCELoss()
        self.bs = args.batch_size

    def forward(self, scores, masks):
        # video-level score is the mean of top 10% scores
        fpn_scores = [x for x in scores]
        sorted_scores = [x.sort(descending=True, dim=1)[0] for x in fpn_scores]
        k_value = [torch.ceil(x.sum(-1) * 0.1).int() for x in masks]
        for i, (score, k) in enumerate(zip(sorted_scores, k_value)):
            ids = torch.arange(0, score.shape[1], device=score.device)
            mask = (ids < k.unsqueeze(-1)).bool()
            score = score * mask
            fpn_scores[i] = score.sum(1) / k
        input = torch.stack(fpn_scores).mean(dim=0)
        target = torch.zeros_like(input)
        target[self.bs :] = 1
        loss = self.criterion(input, target)
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.triplet = nn.TripletMarginLoss(margin)

    def forward(self, contrast_pairs, norm=True):
        n_layers = len(contrast_pairs["ABN_EMB_PRED"])
        loss = 0.0
        for i in range(n_layers):
            anchor_i = contrast_pairs["ABN_EMB_PRED"][i].mean(1)
            positive_i = contrast_pairs["ABN_EMB_PSE"][i].mean(1)
            negetive_i = contrast_pairs["N_EMB"][i].mean(1)

            if norm:
                anchor_i = self.norm(anchor_i)
                positive_i = self.norm(positive_i)
                negetive_i = self.norm(negetive_i)
            loss += self.triplet(anchor_i, positive_i, negetive_i)
        return loss / n_layers

    # def forward(self, contrast_pairs, norm=True):
    #     anchor = torch.cat(contrast_pairs["ABN_EMB_PRED"], dim=1).mean(dim=1)
    #     positive = torch.cat(contrast_pairs["ABN_EMB_PSE"], dim=1).mean(dim=1)
    #     negetive = torch.cat(contrast_pairs["N_EMB"], dim=1).mean(dim=1)

    #     if norm:
    #         anchor = self.norm(anchor)
    #         positive = self.norm(positive)
    #         negetive = self.norm(negetive)
    #     loss = self.triplet(anchor, positive, negetive)
    #     return loss

    def norm(self, data):
        l2 = torch.norm(data, p=2, dim=-1, keepdim=True)
        return torch.div(data, l2)


class TotalLoss(nn.Module):
    def __init__(self, args, num_tasks=2):
        super(TotalLoss, self).__init__()
        params = torch.ones(num_tasks, requires_grad=True)
        self.params = torch.nn.Parameter(params)

        self.pse = PseLoss(args)
        self.cls = ClsLoss(args)
        # self.triplet = TripletLoss()

    def forward(self, scores, logits, masks, contrast_pairs, pseudo_label):
        loss_pse = self.pse(logits, masks, pseudo_label)
        loss_cls = self.cls(scores, masks)
        # loss_trip = self.triplet(contrast_pairs)
        loss_total = self.cal_loss(loss_pse, loss_cls)
        loss_dict = {
            "Loss/PSE": loss_pse.item(),
            "Loss/Cls": loss_cls.item(),
            # "Loss/Tri": loss_trip.item(),
            "Loss/Total": loss_total.item(),
        }
        return loss_total, loss_dict

    def cal_loss(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum
