import torch
import numpy as np
import re
from tqdm import tqdm
import argparse
from collections import defaultdict
import sys

sys.path.append('..')
import clip
from utils import upgrade_resolution_th


DATASET_INFO = {
    "ucfcrime": {
        "train_list": "UCF_Train.list",  # the train video_id list
        "test_list": "UCF_Test.list",  # the test video_id list
        "zip_feats": "ucfcrime_i3d_roc_ng_w16_s16.zip",  # dataset zip path
        "suffix": "Normal",  # the suffix to split normal and abnormal videos
        "window_size": 16,
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
        "train_list": "XDViolence_Train.list",
        "test_list": "XDViolence_Test.list",
        "zip_feats": "xdviolence_i3d_w16_s16.zip",
        "suffix": "label_A",
        "window_size": 16,
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


CLIP_MODELS = {
    "ucfcrime": {
        "RN50x4": {
            "clip_zip_feats": "/data/qianzhangbin/datasets/UCF-Crime/qzb_extract/clip/ucfcrime_clip_RN50x4_5fps.zip",  # clip visual features zip path
            "dump_path": "./UCF_frame_pseudo_RN50x4.npy",
            "template": "a photo of a person during {}.",
        },
        "RN50x16": {
            "clip_zip_feats": "/data/qianzhangbin/datasets/UCF-Crime/qzb_extract/clip/ucfcrime_clip_RN50x16_5fps.zip",
            "dump_path": "./UCF_frame_pseudo_RN50x16.npy",
            "template": "a photo of the person performing {}.",
        },
        "ViT-B/16": {
            "clip_zip_feats": "/data/qianzhangbin/datasets/UCF-Crime/qzb_extract/clip/ucfcrime_clip_ViT-B_16_5fps.zip",
            "dump_path": "./UCF_frame_pseudo_ViT-B_16.npy",
            "template": "a video of a person during {}.",
        },
        "ViT-B/32": {
            "clip_zip_feats": "/data/qianzhangbin/datasets/UCF-Crime/qzb_extract/clip/ucfcrime_clip_ViT-B_32_5fps.zip",
            "dump_path": "./UCF_frame_pseudo_ViT-B_32.npy",
            "template": "a video of the person performing {}.",
        },
    },
    "xdviolence": {
        "RN50x4": {
            "clip_zip_feats": "/data/qianzhangbin/datasets/XD-Violence/qzb_extract/clip/xdviolence_clip_RN50x4_4fps.zip",  # clip visual features zip path
            "dump_path": "./XD_frame_pseudo_label_RN50x4.npy",
            "template": "a demonstration of a person practicing {}.",
        },
        "ViT-B/16": {
            "clip_zip_feats": "/data/qianzhangbin/datasets/XD-Violence/qzb_extract/clip/xdviolence_clip_ViT-B_16_4fps.zip",
            "dump_path": "./XD_frame_pseudo_label_ViT-B_16.npy",
            "template": "a video of a person performing {}.",
        },
        "ViT-B/32": {
            "clip_zip_feats": "/data/qianzhangbin/datasets/XD-Violence/qzb_extract/clip/xdviolence_clip_ViT-B_32_4fps.zip",
            "dump_path": "./XD_frame_pseudo_label_ViT-B_32.npy",
            "template": "a demonstration of a person practicing {}.",
        },
    },
}


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for clip_name, clip_info in CLIP_MODELS[args.dataset_name].items():
        # initialize dataloader
        dataset_info = DATASET_INFO[args.dataset_name]
        train_vids = set(x.strip()[:-3] for x in open(dataset_info["train_list"]) if dataset_info["suffix"] not in x)
        class_names = list(dataset_info["class_map"].values())
        print("class names: ", class_names)
        dataloader = np.load(clip_info["clip_zip_feats"])
        zip_feats = np.load(dataset_info["zip_feats"])

        # initialize model
        model, _ = clip.load(clip_name, device)
        prompt = clip_info["template"]
        text_inputs = [clip.tokenize(prompt.format(c)) for c in class_names]

        # dump pseudo labels
        print(f"dump pseudo labels to {clip_info['dump_path']} by {clip_name}, the prompt is: {prompt}")
        dump(args, train_vids, dataloader, zip_feats, model, text_inputs, dataset_info, clip_info, device)
        print(f"dump pseudo labels for {clip_name} is done! \n")


@torch.no_grad()
def dump(args, video_ids, dataloader, zip_feats, model, text_inputs, dataset_info, clip_info, device):
    """
    dump frame-level pseudo labels by clip model
    """
    model.eval()
    text_inputs = torch.cat(text_inputs).to(device)
    class_map = dataset_info["class_map"]
    framePseudoLabel = {}
    for video_id in tqdm(video_ids):
        image_features = torch.from_numpy(dataloader[f"{video_id}_clip"]).float().to(device, non_blocking=True)
        text_features = model.encode_text(text_inputs).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = model.logit_scale.exp().to(device, dtype=image_features.dtype)
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

        pred = torch.zeros(similaritys.shape[0], device=device)
        for idx in vid_class_idx:
            pred += similaritys[:, idx]

        # upgrade resolution
        num_frame = zip_feats[f"{video_id}__0"].shape[0] * dataset_info['window_size']
        scale = num_frame / pred.shape[0]
        pred = upgrade_resolution_th(pred.unsqueeze(0), scale, mode="linear").cpu().numpy()
        if num_frame <= len(pred):
            pred = pred[:num_frame]
        else:
            pred = np.concatenate((pred, np.full(num_frame - pred.shape[0], pred[-1])))
        # Normalize
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        framePseudoLabel[video_id] = pred.astype(np.float32)
    np.save(clip_info['dump_path'], framePseudoLabel)


def ensemble(args):
    """
    ensemble pseudo labels
    """
    pseudo_label_list = []
    for _, clip_info in CLIP_MODELS[args.dataset_name].items():
        pseudo_label = np.load(clip_info['dump_path'], allow_pickle=True).item()
        pseudo_label_list.append(pseudo_label)
    merged = defaultdict(list)
    for d in pseudo_label_list:
        for key, value in d.items():
            merged[key].append(value)
    result = {}
    for key, values in merged.items():
        pred = sum(values) / len(values)
        pred = (pred - pred.min()) / (pred.max() - pred.min())  # Normalize
        result[key] = pred
    if args.dataset_name == "ucfcrime":
        np.save("UCF_frame_pseudo_label.npy", result)
    elif args.dataset_name == "xdviolence":
        np.save("XD_frame_pseudo_label.npy", result)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Pseudo Label Evaluation")
    parser.add_argument('--dataset_name', type=str, required=True, help="dataset name ['ucfcrime', 'xdviolence']")
    args = parser.parse_args()

    main(args)
    ensemble(args)
