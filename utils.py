import sys
import numpy as np
import torch
import logging
import random
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from torch import inf
import os
import torch.nn.functional as F
import csv
import requests
from blocks import MaskedConv1D, Scale, AffineDropPath, LayerNorm
import functools


def setup_device(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        # training: disable cudnn benchmark to ensure the reproducibility
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)


@functools.lru_cache()
def setup_logging(logger_name, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(asctime)s [%(levelname)s] %(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode="a")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def setup_dataset(args):
    assert os.path.isfile(args.zip_feats), f"{args.zip_feats} is not exists"
    # Choose dataset
    if "ucfcrime" in str(args.zip_feats):
        args.dataset_name = "ucfcrime"
        if args.tencrop:
            args.test_batch_size = 10
    elif "xdviolence" in str(args.zip_feats):
        args.dataset_name = "xdviolence"
        if args.tencrop:
            args.test_batch_size = 5
    else:
        raise RuntimeError(f"{args.zip_feats} is not supported")


def update_args_(args, params):
    """updates args in-place"""
    dargs = vars(args)
    dargs.update(params)


def cal_false_alarm(scores, labels, threshold=0.5):
    scores = np.array([1 if score > threshold else 0 for score in scores], dtype=float)
    fp = np.sum(scores * (1 - labels))
    return fp / np.sum(1 - labels)


def cal_false_alarm_th(scores, labels, threshold=0.5):
    scores = (scores > threshold).float()
    fp = torch.sum(scores * (1 - labels))
    return fp / torch.sum(1 - labels)


def upgrade_resolution_th(arr, scale, mode="nearest"):
    """Upgrade resolution by interpolation, support 'nearest', 'linear', 'area', 'nearest-exact, modes"""
    if len(arr.shape) == 2:
        b, t = arr.shape
        arr = arr.view(b, 1, t)
    f = F.interpolate(arr, scale_factor=scale, mode=mode)
    return f.squeeze()


def restore_checkpoint(args, model_to_load, restore_file):
    state_dict = torch.load(restore_file, map_location=torch.device("cpu"))["model_state_dict"]

    own_state = model_to_load.state_dict()
    for name, param in state_dict.items():
        name = name.replace("module.", "")
        if name not in own_state:
            print(f"Skipped: {name}")
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
            # print(f"Successfully loaded: {name}")
        except:
            pass
            print(f"Part load failed: {name}")


def build_optimizer_scheduler(args, model, num_total_steps):
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm)

    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            if "txt_model" in fpn:
                continue
            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith("scale") and isinstance(m, (Scale, AffineDropPath)):
                # corner case of our scale layer
                no_decay.add(fpn)
            elif pn.endswith("rel_pe"):
                # corner case for relative position encoding
                no_decay.add(fpn)

    # create the pytorch optimizer object
    param_dict = {pn: p for pn, p in model.named_parameters()}
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": args.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.learning_rate)

    # bulid scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=num_total_steps)
    if args.lr_scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * num_total_steps), num_training_steps=num_total_steps)
    elif args.lr_scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * num_total_steps))

    return optimizer, scheduler


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def write_csv(filename, data):
    # check if csv file exists
    if not os.path.isfile(filename):
        with open(filename, "w") as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            writer.writeheader()
            writer.writerow(data)
    else:
        has_headings = False
        # check if csv file has header
        with open(filename, "r") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != None:
                has_headings = True

        # write data into it
        with open(filename, "a+") as file:
            if not has_headings:
                writer = csv.DictWriter(file, fieldnames=data.keys())
                writer.writeheader()
            else:
                writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
            writer.writerow(data)


def write_notion(database_id, token, data, proxies={"http": "", "https": ""}):
    url = "https://api.notion.com/v1/databases/" + database_id
    headers = {"accept": "application/json", "Notion-Version": "2022-06-28", "Authorization": "Bearer " + token}
    response = requests.get(url, headers=headers, proxies=proxies)

    # pack data to push
    data_dict = {}
    for key in response.json()["properties"]:
        if key == "save_path":
            value = data[key].split("/")[-1]
            data_dict[key] = {"title": [{"text": {"content": value}}]}
        else:
            try:
                data_dict[key] = {"rich_text": [{"text": {"content": f"{data[key]}"}}]}
            except:
                pass
    # push to notion database
    create_url = "https://api.notion.com/v1/pages"
    payload = {"parent": {"database_id": database_id}, "properties": data_dict}
    res = requests.post(create_url, headers=headers, json=payload, proxies=proxies)
    if res.status_code == 200:
        print("Push to Notion success!")
    else:
        print("Push to Notion failed!")
