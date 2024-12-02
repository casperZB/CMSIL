import torch
from pathlib import Path
import numpy as np
import re
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve, precision_recall_curve, classification_report, accuracy_score
import argparse
import sys

sys.path.append('..')
import clip
from utils import write_csv, setup_device, upgrade_resolution_th


DATASET_INFO = {
    "ucfcrime": {
        "train_list": "UCF_Train.list",  # the train video_id list
        "test_list": "UCF_Test.list",  # the test video_id list
        "gt_frames": "/data/qianzhangbin/repo/vadclip/data/UCF_frame_label.npy",  # the frame-level label
        "suffix": "Normal",  # the suffix to split normal and abnormal videos
        "class_map": {
            "Abuse": "Abuse",
            "Arrest": "Arrest",
            "Arson": "Arson",
            "Assault": "Assault",
            "Burglary": "Burglary",
            "Explosion": "Explosion",
            "Fighting": "Fighting",
            "RoadAccidents": "RoadAccidents",
            "Robbery": "Robbery",
            "Shooting": "Shooting",
            "Shoplifting": "Shoplifting",
            "Stealing": "Stealing",
            "Vandalism": "Vandalism",
        },
    },
    "xdviolence": {
        "train_list": "XDViolence_Test.list",
        "test_list": "XDViolence_Test.list",
        "gt_frames": "/data/qianzhangbin/repo/vadclip/data/XD_frame_label.npy",
        "suffix": "label_A",
        "class_map": {
            "B1": "Fighting",
            "B2": "Shooting",
            "B4": "Riot",
            "B5": "Abuse",
            "B6": "Car accident",
            "G": "Explosion",
        },
    },
}

templates = [
    "a photo of a person {}.",
    "a video of a person {}.",
    "a example of a person {}.",
    "a demonstration of a person {}.",
    "a photo of the person {}.",
    "a video of the person {}.",
    "a example of the person {}.",
    "a demonstration of the person {}.",
    "a photo of a person using {}.",
    "a video of a person using {}.",
    "a example of a person using {}.",
    "a demonstration of a person using {}.",
    "a photo of the person using {}.",
    "a video of the person using {}.",
    "a example of the person using {}.",
    "a demonstration of the person using {}.",
    "a photo of a person doing {}.",
    "a video of a person doing {}.",
    "a example of a person doing {}.",
    "a demonstration of a person doing {}.",
    "a photo of the person doing {}.",
    "a video of the person doing {}.",
    "a example of the person doing {}.",
    "a demonstration of the person doing {}.",
    "a photo of a person during {}.",
    "a video of a person during {}.",
    "a example of a person during {}.",
    "a demonstration of a person during {}.",
    "a photo of the person during {}.",
    "a video of the person during {}.",
    "a example of the person during {}.",
    "a demonstration of the person during {}.",
    "a photo of a person performing {}.",
    "a video of a person performing {}.",
    "a example of a person performing {}.",
    "a demonstration of a person performing {}.",
    "a photo of the person performing {}.",
    "a video of the person performing {}.",
    "a example of the person performing {}.",
    "a demonstration of the person performing {}.",
    "a photo of a person practicing {}.",
    "a video of a person practicing {}.",
    "a example of a person practicing {}.",
    "a demonstration of a person practicing {}.",
    "a photo of the person practicing {}.",
    "a video of the person practicing {}.",
    "a example of the person practicing {}.",
    "a demonstration of the person practicing {}.",
]


def main(args):
    # initialize dataloader
    info = DATASET_INFO[args.dataset_name]
    dataloader = np.load(args.clip_zip_feats)
    train_vids = set(x.strip()[:-3] for x in open(info["train_list"]) if info["suffix"] not in x)
    class_names = list(info["class_map"].values())
    print("class names: ", class_names)
    gt_frames = np.load(info["gt_frames"], allow_pickle=True).item()

    # initialize model
    model, _ = clip.load(args.clip, args.device)

    # dump pseudo labels and evaluate different prompts
    file_name = f"Result-{Path(args.clip_zip_feats).stem}-win{args.window_size}.csv"
    for f in templates:
        text_inputs = [clip.tokenize(f.format(c)) for c in class_names]

        # dump pseudo labels to disk
        auc_test, ap_test = dump(args, train_vids, dataloader, model, text_inputs, info["class_map"], gt_frames)
        results = {
            "prompt": f,
            "AUC_test": round(auc_test, 3),
            "AP_test": round(ap_test, 3),
        }
        write_csv(file_name, results)


@torch.no_grad()
def dump(args, video_ids, dataloader, model, text_inputs, class_map, gt_frames):
    """
    dump frame-level pseudo labels by clip model
    """
    model.eval()
    text_inputs = torch.cat(text_inputs).to(args.device)
    framePseudoLabel = {}
    cls_pred, cls_true = np.empty(0), np.empty(0)
    for video_id in tqdm(video_ids):
        image_features = torch.from_numpy(dataloader[f"{video_id}_clip"]).float().to(args.device, non_blocking=True)
        text_features = model.encode_text(text_inputs).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = model.logit_scale.exp().to(args.device, dtype=image_features.dtype)
        similaritys = (logit_scale * image_features @ text_features.T).softmax(dim=-1)

        class_idx = list(class_map.keys())
        if args.dataset_name == "ucfcrime":
            vid_class = re.sub(r"\d+", "", video_id.split("_")[0])
            vid_class_idx = [class_idx.index(vid_class)]
        elif args.dataset_name == "xdviolence":
            vid_class = video_id.split("label_")[-1].split("-")
            vid_class_idx = [class_idx.index(c) for c in vid_class if c in class_idx]
        else:
            raise NotImplementedError
        assert len(vid_class_idx) > 0, f"video {video_id} has no class label"

        pred = torch.zeros(similaritys.shape[0], device=args.device)
        for idx in vid_class_idx:
            pred += similaritys[:, idx]

        # upgrade resolution
        gt_frame = gt_frames[video_id]
        scale = gt_frame.shape[0] / pred.shape[0]
        pred = upgrade_resolution_th(pred.unsqueeze(0), scale, mode="linear").cpu().numpy()
        if len(gt_frame) <= len(pred):
            pred = pred[: len(gt_frame)]
        else:
            pred = np.concatenate((pred, np.full(len(gt_frame) - len(pred), pred[-1])))
        # Normalize
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        framePseudoLabel[video_id] = pred

        cls_pred = np.concatenate((cls_pred, pred))
        cls_true = np.concatenate((cls_true, gt_frame))

    # Compute Area Under the Receiver Operating Characteristic Curve
    fpr, tpr, _ = roc_curve(cls_true, cls_pred)
    auc_score = auc(fpr, tpr)

    # Compute Average Precision score
    precision, recall, _ = precision_recall_curve(cls_true, cls_pred)
    ap_score = auc(recall, precision)

    return auc_score, ap_score


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Pseudo Label Evaluation")
    parser.add_argument('--clip', type=str, default='RN50', help="A variant of CLIP. ['ViT-B/16', 'ViT-B/32', 'RN50x16', 'RN50x4', 'RN101', 'RN50'] are supported.")
    parser.add_argument('--clip_zip_feats', type=str, default='ucfcrime_clip_RN50_5fps.zip', help="clip visual features zip path")
    parser.add_argument("--dataset_name", default="ucfcrime", help="dataset name ['ucfcrime', 'xdviolence']")
    parser.add_argument("--window_size", type=int, default=1, help="The number of frames from which to extract features (or window size")
    args = parser.parse_args()
    setup_device(args)
    print(("\n#### CONFIG ####\n" + "".join(f"{k}: {v}\n" for k, v in vars(args).items())))
    main(args)
