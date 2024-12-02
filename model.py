import torch
from torch import nn
from torch.nn import functional as F
from backbones import ConvTransformerBackbone, FPNIdentity, ClsHead
import math
import random

featuredim = {
    "rgb": 1024,
    "flow": 1024,
    "both": 2048,
}


class VADTransformer(nn.Module):
    """
    Transformer based model for video anomaly detection
    """

    def __init__(self, args):
        super(VADTransformer, self).__init__()
        self.bs = args.batch_size
        self.n_layers = args.arch[2]
        self.mha_win_size = [args.n_mha_win_size] * self.n_layers
        self.max_seq_len = args.max_seq_len
        self.backbone = ConvTransformerBackbone(
            n_in=featuredim[args.modality],
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_embd_ks=3,
            max_len=args.max_seq_len,
            arch=(args.arch[0], args.arch[1], args.arch[2] - 1),
            mha_win_size=self.mha_win_size,
            scale_factor=args.scale_factor,
            with_ln=True,
        )

        self.neck = FPNIdentity(
            in_channels=[args.n_embd] * self.n_layers,
            out_channel=args.n_embd,
            scale_factor=args.scale_factor,
            start_level=0,
            with_ln=True,
            drop_rate=args.dropout,
            se_enhance=args.se_ratio,
        )

        self.cls_head = ClsHead(
            input_dim=args.n_embd,
            feat_dim=args.n_embd,
            num_classes=1,
            kernel_size=3,
            with_ln=True,
        )

    def forward(self, inputs, is_training=False):
        # forward the network (backbone -> neck -> heads)
        feats = inputs["feats"].permute(0, 2, 1)  # (B, C, T)
        masks = inputs["masks"].unsqueeze(1)  # (B, 1, T)
        feats, masks = self.backbone(feats, masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        logits = self.cls_head(fpn_feats, fpn_masks)  # (B, cls, T)

        # output
        logits = [x.squeeze() for x in logits]  # (B, 1, T) -> (B, T)
        fpn_masks = [x.squeeze() for x in fpn_masks]  # (B, 1, T) -> (B, T)
        scores = [x.sigmoid() * y for x, y in zip(logits, fpn_masks)]  # (B, T)

        if is_training:
            contrast_pairs = None
            pseudo_label = inputs["pseudo_label"]
            fpn_feats = [f.permute(0, 2, 1) for f in fpn_feats]  # (B, T, C)

            ABN_EMB_PRED, ABN_EMB_PSE, N_EMB = tuple(), tuple(), tuple()
            for i, mask in enumerate(fpn_masks):
                # select representative normal feature
                k_normal = math.ceil(mask[0 : self.bs].sum(-1).float().mean().item() * random.uniform(0.2, 0.4))
                N_EMB += (self.select_topk_embeddings(scores[i][0 : self.bs], fpn_feats[i][0 : self.bs], k_normal),)

                # select top 10% representative abnormal feature from predict
                k_abnormal = math.ceil(mask[self.bs :].sum(-1).float().mean().item() * random.uniform(0.1, 0.3))
                ABN_EMB_PRED += (self.select_topk_embeddings(scores[i][self.bs :], fpn_feats[i][self.bs :], k_abnormal),)
                # select top 10% representative abnormal feature from pseudo label
                pse_label_i = torch.max_pool1d(pseudo_label[self.bs :], kernel_size=2**i).float()
                ABN_EMB_PSE += (self.select_topk_embeddings(pse_label_i, fpn_feats[i][self.bs :], k_abnormal),)

            contrast_pairs = {"ABN_EMB_PRED": ABN_EMB_PRED, "ABN_EMB_PSE": ABN_EMB_PSE, "N_EMB": N_EMB}
            return scores, logits, fpn_masks, contrast_pairs
        return scores

    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1)
        idx_topk = idx_DESC[:, :k]
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings


if __name__ == "__main__":
    # Choose config params for different datasets
    import sys
    import importlib
    from utils import setup_dataset

    for arg in sys.argv:
        if "ucfcrime" in str(arg):
            config_module = importlib.import_module("config.ucfcrime_cfg")
        elif "xdviolence" in str(arg):
            config_module = importlib.import_module("config.xdviolence_cfg")
    parser = config_module.parse_args()
    args = parser.parse_args()
    model = VADTransformer(args)
    # print(model)

    # inputs = torch.randn(1, 256, 1024)
    # mask = torch.ones(1, 1, 256)
    # out = model(inputs, mask)

    # from torchinfo import summary

    # summary(model, input_data=(torch.randn(1, 384, 1024), torch.ones(1, 384)))
