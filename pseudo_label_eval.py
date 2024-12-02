import torch
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
import clip
import re
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve, precision_recall_curve, classification_report, accuracy_score
from utils import write_csv, setup_device
import time
import argparse

DATASET_INFO = {
    "ucfcrime": {
        "train_list": "data/UCF_Train.list",  # the train video_id list
        "test_list": "data/UCF_Test.list",  # the test video_id list
        "gt_frames": "data/UCF_frame_label.npy",  # the frame-level label
        "root_path": "data/UCF_Crime/Videos",  # the raw video path
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
        "train_list": "data/XDViolence_Train.list",
        "test_list": "data/XDViolence_Test.list",
        "gt_frames": "data/XD_frame_label.npy",
        "root_path": "data/XD_Violence/Videos",
        "split_idx": 1905,
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

templates = [
    "a photo of a person doing {}.",
]


def main(args):
    # initialize dataloader
    info = DATASET_INFO[args.dataset_name]
    dataloader = np.load(args.zip_feats)
    train_vids = set(x.strip()[:-3] for x in open(info["train_list"]) if info["suffix"] not in x)
    test_vids = set(x.strip()[:-3] for x in open(info["test_list"]) if info["suffix"] not in x)
    class_names = list(info["class_map"].values())
    print("class names: ", class_names)
    gt_frames = np.load(info["gt_frames"], allow_pickle=True).item()

    # initialize model
    model, _ = clip.load(args.clip, args.device)

    # dump pseudo labels and evaluate different prompts
    file_name = f"Result-{Path(args.zip_feats).stem}-win{args.window_size}.csv"
    for f in templates:
        text_inputs = [clip.tokenize(f.format(c)) for c in class_names]

        # dump pseudo labels to disk
        if args.dump_path is not None:
            dump(args, train_vids, dataloader, model, text_inputs, info["class_map"])

        if args.eval:
            # evaluate different prompts
            auc_train, ap_train, acc_train = eval_pse(args, train_vids, dataloader, model, text_inputs, gt_frames, info["class_map"])
            auc_test, ap_test, acc_test = eval_pse(args, test_vids, dataloader, model, text_inputs, gt_frames, info["class_map"])

            results = {
                "prompt": f,
                "AUC_trian": auc_train,
                "AUC_test": auc_test,
                "AP_train": ap_train,
                "AP_test": ap_test,
                "ACC_train": acc_train,
                "ACC_test": acc_test,
                "Mean": round((auc_train + auc_test + ap_train + ap_test + acc_train + acc_test) / 6, 3),
            }
            write_csv(file_name, results)


@torch.no_grad()
def eval_pse(args, video_ids, dataloader, model, text_inputs, gt_frames, class_map, threshold=0.5):
    """
    evaluate pseudo labels performance, support frame-level, segment-level
    """
    model.eval()
    text_inputs = torch.cat(text_inputs).to(args.device)
    vids, pred_labels, gt_labels = [], [], []
    for video_id in tqdm(video_ids):
        image_features = torch.from_numpy(dataloader[video_id]).to(args.device, non_blocking=True)
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
            vid_class_idx = [class_idx.index(c) for c in vid_class]
        else:
            raise NotImplementedError
        assert len(vid_class_idx) > 0, f"video {video_id} has no class label"

        pred = torch.zeros(similaritys.shape[0], device=args.device)
        for idx in vid_class_idx:
            pred += similaritys[:, idx]
        pred = pred.cpu().numpy()  # (T, )
        pred = (pred - pred.min()) / (pred.max() - pred.min())  # Normalize
        vids.append(video_id)
        pred_labels.append(pred)
        gt_labels.append(gt_frames[video_id])

    # eval
    gts, preds = np.zeros(0), np.zeros(0)
    cls_true, cls_pred = [], []
    for vid, pred, gt in zip(vids, pred_labels, gt_labels):
        # for segment-level: using sliding window to get segment-level prediction and gt
        win_sz = args.window_size
        pred = np.array([pred[i : i + win_sz].max() for i in range(0, len(pred) - win_sz + 1, win_sz)])
        gt = np.array([gt[i : i + win_sz].max() for i in range(0, len(gt) - win_sz + 1, win_sz)])

        preds = np.concatenate((preds, pred))
        gts = np.concatenate((gts, gt))

        # get video level prediction
        cls_pred_i = np.mean(np.partition(pred, -3)[-3:])
        cls_pred.append(1 if cls_pred_i > threshold else 0)
        cls_true.append(1)

    # Compute Area Under the Receiver Operating Characteristic Curve
    fpr, tpr, _ = roc_curve(gts, preds)
    auc_score = auc(fpr, tpr)

    # Compute Average Precision score
    precision, recall, _ = precision_recall_curve(gts, preds)
    ap_score = auc(recall, precision)

    accuracy = accuracy_score(cls_true, cls_pred)

    return round(auc_score, 3), round(ap_score, 3), round(accuracy, 3)


@torch.no_grad()
def dump(args, video_ids, dataloader, model, text_inputs, class_map):
    """
    dump frame-level pseudo labels by clip model
    """
    model.eval()
    text_inputs = torch.cat(text_inputs).to(args.device)
    framePseudoLabel = {}
    for video_id in tqdm(video_ids):
        image_features = torch.from_numpy(dataloader[video_id]).to(args.device, non_blocking=True)
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
        pred = pred.cpu().numpy()  # (T, )
        pred = (pred - pred.min()) / (pred.max() - pred.min())  # Normalize
        framePseudoLabel[video_id] = pred
    np.save(args.dump_path, framePseudoLabel)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Make Pseudo Label")
    parser.add_argument('--clip', type=str, required=True, default='ViT-B/16', help="clip model type")
    parser.add_argument("--zip_feats", type=str, required=True, default="data/ucfcrime_clip_vit-B16.zip", help="clip visual features zip path")
    parser.add_argument("--dataset_name", type=str, required=True, default="ucfcrime", help="dataset name ['ucfcrime', 'xdviolence']")
    parser.add_argument("--window_size", type=int, default=1, help="The number of frames from which to extract features (or window size")
    parser.add_argument("--dump_path", type=str, default=None, help="the pseudo label path to dump")
    parser.add_argument("--eval", action="store_true", help="whether to evaluate pseudo labels performance")
    args = parser.parse_args()
    setup_device(args)
    print(("\n#### CONFIG ####\n" + "".join(f"{k}: {v}\n" for k, v in vars(args).items())))
    main(args)
