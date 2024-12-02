import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import re

DATASET_INFO = {
    "ucfcrime": {
        "train_list": "data/UCF_Train.list",  # the train video_id list
        "test_list": "data/UCF_Test.list",  # the test video_id list
        "gt_frames": "data/UCF_frame_label.npy",  # the frame-level label
        "gt_videos": "data/UCF_video_label.npy",  # the frame-level label
        "split_idx": 8100,  # the index to split normal and abnormal videos
    },
    "xdviolence": {
        "train_list": "data/XDViolence_Train.list",
        "test_list": "data/XDViolence_Test.list",
        "gt_frames": "data/XD_frame_label.npy",
        "gt_videos": "data/XD_video_label.npy",  # the frame-level label
        "split_idx": 9525,
    },
}


class VADDataset(Dataset):
    def __init__(self, args, is_normal=True, test_mode=False):
        self.is_normal = is_normal
        self.test_mode = test_mode
        self.tencrop = args.tencrop
        self.modality = args.modality
        self.max_seq_len = args.max_seq_len
        self.dataset_name = args.dataset_name
        self.window_size = args.window_size
        self.zip_feats = args.zip_feats
        self.max_div_factor = 1
        if self.test_mode:
            self._get_max_div_factor(args)
        self._parse()
        self.pseudo_label = np.load(args.pseudo_label, allow_pickle=True).item()

        # lazy initialization for zip_handler to avoid multiprocessing-reading error
        self.num_workers = args.num_workers
        if self.num_workers > 0:
            self.handles = [None for _ in range(args.num_workers)]
        else:
            self.handles = np.load(self.zip_feats)

    def __getitem__(self, index):
        # padding features for batch training
        vid, features, masks, pseudo_label, num_segments = self.load_feats(self.modality, index)

        # return a data dict
        data_dict = {
            "video_id": vid,
            "feats": features,
            "num_segments": num_segments,
            "masks": masks,
        }

        if not self.test_mode:
            data_dict["pseudo_label"] = pseudo_label
        return data_dict

    def load_feats(self, modality, index: int) -> tuple:
        """load features from zipfile"""
        vid = self.video_ids[index]
        if self.num_workers > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handles[worker_id] is None:
                self.handles[worker_id] = np.load(self.zip_feats)
            handle = self.handles[worker_id]
        else:
            handle = self.handles
        raw_feats = handle[vid]

        # process features for batch training
        features, mask, pseudo_label = self.batch_input(raw_feats, vid)
        num_segments = torch.LongTensor(torch.sum(mask))
        return vid, features, mask, pseudo_label, num_segments

    def batch_input(self, feat, vid):
        """process features for batch training
        for train mode: randomly truncate the features if the length is larger than the max_seq_len
        for test mode: if the number of frames exceeds the limitation, pad the features to the next divisible size of max_div_factor
        """
        vid = vid[:-3]
        t, f = feat.shape
        max_seq_len = self.max_seq_len
        if self.test_mode and t > self.max_seq_len:
            # pad the features to the next divisible size of max_div_factor
            max_seq_len = (t + (self.max_div_factor - 1)) // self.max_div_factor * self.max_div_factor

        pseudo_label = None
        if self.test_mode:
            feat = np.pad(feat, ((0, max_seq_len - t), (0, 0)), mode="constant")
            mask = torch.ones((max_seq_len), dtype=torch.bool)
            mask[t:] = False
        else:
            pse_label = torch.zeros(t)
            if not self.is_normal:
                pse_label = torch.from_numpy(self.pseudo_label[vid]).reshape(1, -1)
                pse_label = torch.max_pool1d(pse_label, self.window_size).squeeze()
                pse_label = pse_label[:t]
            feat, mask, pseudo_label = self.truncate_feats(feat, vid, pse_label)
        return torch.from_numpy(feat), mask, pseudo_label

    def truncate_feats(self, feat, vid, pseudo_label, trunc_thresh=0.5, max_num_trials=200):
        t, f = feat.shape
        mask = torch.ones((self.max_seq_len), dtype=torch.bool)

        # if the sequence len is smaller than the max_seq_len, pad it
        if t <= self.max_seq_len:
            padding_size = self.max_seq_len - t
            feat = np.pad(feat, ((0, padding_size), (0, 0)), mode="constant")
            mask[t:] = False
            pseudo_label = torch.cat([pseudo_label, torch.zeros(padding_size)])
            return feat, mask, pseudo_label

        # randomly truncate the features if the length is larger than the max_seq_len
        pseudo_info = torch.where(pseudo_label > 0.5, torch.tensor([1]), torch.tensor([0])).tolist()
        valid_segs = [(m.start(), (m.end() - 1)) for m in re.finditer("1{5,}", "".join(map(str, pseudo_info)))]  # continuous 5 abnormal segments
        valid_segs = torch.as_tensor(valid_segs, dtype=torch.float32)

        # try a few times till a valid truncation with at least one abnormal segment is found
        for _ in range(max_num_trials):
            # sample a random truncation of the video feats
            st = random.randint(0, t - self.max_seq_len)
            ed = st + self.max_seq_len
            window = torch.as_tensor([st, ed], dtype=torch.float32)

            # compute the intersection between the sampled window and all segments
            num_segs = len(valid_segs)
            if num_segs == 0:
                return feat[st:ed], mask, pseudo_label[st:ed]  # for normal segment, just return the random one
            window = window[None].repeat(num_segs, 1)
            left = torch.maximum(window[:, 0], valid_segs[:, 0])
            right = torch.minimum(window[:, 1], valid_segs[:, 1])
            inter = (right - left).clamp(min=0)
            area_segs = torch.abs(valid_segs[:, 1] - valid_segs[:, 0]).clamp(min=1e-4)
            inter_ratio = inter / area_segs

            # only select those segments over the thresh
            seg_idx = inter_ratio >= trunc_thresh
            if seg_idx.sum().item() > 0:
                return feat[st:ed], mask, pseudo_label[st:ed]
        # if no valid truncation is found, just return the random one
        return feat[st:ed], mask, pseudo_label[st:ed]

    def __len__(self):
        return len(self.video_ids)

    def _parse(self):
        if self.dataset_name not in DATASET_INFO:
            raise RuntimeError(f"{self.zip_feats} is not supported")

        info = DATASET_INFO[self.dataset_name]
        self.gt_frames = np.load(info["gt_frames"], allow_pickle=True).item()
        self.gt_videos = np.load(info["gt_videos"], allow_pickle=True).item()
        self.list_file = info["test_list" if self.test_mode else "train_list"]
        self.video_ids = [x.strip() for x in open(self.list_file, encoding="utf-8")]
        if self.test_mode:
            if not self.tencrop:
                # only use center crop for testing
                self.video_ids = [x for x in self.video_ids if x.endswith("__4")]
        else:
            # split abnormal and normal videos for training
            self.video_ids = self.video_ids[info["split_idx"] :] if self.is_normal else self.video_ids[: info["split_idx"]]

    def _get_max_div_factor(self, args):
        n_layers = args.arch[2]
        fpn_strides = [args.scale_factor**i for i in range(0, n_layers)]
        mha_win_size = [args.n_mha_win_size] * n_layers
        for s, w in zip(fpn_strides, mha_win_size):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert args.max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride and window size in test mode"
            if self.max_div_factor < stride:
                self.max_div_factor = stride


def create_dataloaders(args):
    # train dataset for normal data
    train_nloader = DataLoader(
        VADDataset(args, test_mode=False, is_normal=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # train dataset for abnormal data
    train_aloader = DataLoader(
        VADDataset(args, test_mode=False, is_normal=False),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # test dataset
    test_loader = DataLoader(
        VADDataset(args, test_mode=True),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_nloader, train_aloader, test_loader


if __name__ == "__main__":
    # Choose config params for different datasets
    import sys
    import importlib
    from utils import setup_dataset

    config_module = importlib.import_module("config.ucfcrime_cfg")
    for arg in sys.argv:
        if "xdviolence" in str(arg):
            config_module = importlib.import_module("config.xdviolence_cfg")
    parser = config_module.parse_args()
    args = parser.parse_args()
    setup_dataset(args)

    args.num_workers = 0
    train_nloader, train_aloader, test_loader = create_dataloaders(args)

    print(f"training dataset for normal data length = {len(train_nloader)}")
    for batch in train_nloader:
        print(batch["video_id"])
        print(batch["feats"].shape)
        print(batch["mask"])
        print(batch["mask"].shape)
        break

    print(f"training dataset for abnormal data length = {len(train_aloader)}")
    for batch in train_aloader:
        print(batch["video_id"])
        print(batch["feats"].shape)
        print(batch["mask"])
        print(batch["mask"].shape)
        break

    print(f"testing dataset length = {len(test_loader)}")
    for batch in test_loader:
        print(batch["video_id"])
        print(batch["feats"].shape)
        print(batch["mask"])
        print(batch["mask"].shape)
        break
