import sys
import os
import torch
from dataset import create_dataloaders
from utils import *
import time
from model import VADTransformer
from tqdm import tqdm
from losses import TotalLoss
from torch.cuda.amp import autocast
import optuna
import importlib
from eval import eval_one_step


def train(args, trial=None):
    setup_seed(args)
    args.arch = args.arch if isinstance(args.arch, list) else [int(x) for x in args.arch.split("_")]
    args.save_path = f"./save/train_{time.strftime('%m%d_%H%M%S', time.localtime())}"
    os.makedirs(f"{args.save_path}/ckpt", exist_ok=True)
    logger = setup_logging(f"{args.save_path}/{args.save_path.split('/')[-1]}.log")
    logger.info("\n#### CONFIG ####\n%s", "".join(f"{k}: {v}\n" for k, v in vars(args).items()))

    # 1. load data
    train_nloader, train_aloader, test_loader = create_dataloaders(args)

    # 2. load model and optimizers
    model = VADTransformer(args)
    if args.ckpt_file is not None:
        restore_checkpoint(args, model, args.ckpt_file)
    if args.device == "cuda":
        model = torch.nn.parallel.DataParallel(model.to(args.device, non_blocking=True))
    optimizer, scheduler = build_optimizer_scheduler(args, model, args.max_step)
    criterion = TotalLoss(args)
    scaler = NativeScalerWithGradNormCount() if args.amp else None

    # 3. training
    best_score = -1
    early_stop_count = 0
    best_result = {}
    validate_freq = len(train_aloader) // 10
    print_step = args.max_step // 15
    losses = []
    for step in tqdm(range(1, args.max_step + 1), total=args.max_step, dynamic_ncols=True):
        if early_stop_count >= 15:
            break
        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)
        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)
            losses.clear()
        loss_dict = train_one_step(args, loadern_iter, loadera_iter, model, optimizer, scheduler, scaler, criterion)
        losses.append(loss_dict)

        # 4. validation
        if step % validate_freq == 0 and step >= print_step:
            early_stop_count += 1
            eval_results = eval_one_step(args, test_loader, model)
            loss = {}
            for key in losses[0].keys():
                loss[key] = round((sum(d[key] for d in losses) / len(losses)), 4)
            # save checkpoint
            improve_str = ""
            score = eval_results[args.eval_metric]

            if trial is not None:
                trial.report(score, step)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if score > best_score:
                early_stop_count = 0
                best_score = score
                if best_score > args.save_score:
                    torch.save(
                        {
                            "args": args,
                            "results": eval_results,
                            "model_state_dict": model.state_dict(),
                        },
                        f"{args.save_path}/ckpt/best-i3d.pth",
                    )
                    improve_str = "Saved"
                    best_result = eval_results
                # print log
                loss_str = "".join(f" {k}:{v:2.4} |" for k, v in loss.items())
                eval_str = "".join(f" {k}:{v:2.4} " for k, v in eval_results.items() if k not in ["frame_level_report", "video_level_report"])
                logger.info("Step %d | NCrop:%s|%s %s", step, eval_str, loss_str, improve_str)

    if best_result:
        save_data = {**vars(args), **{k: v for k, v in best_result.items() if k not in ["frame_level_report", "video_level_report"]}}
        write_csv(f"Result-{time.strftime('%Y-%m-%d')}.csv", save_data)
        logger.info("classification report for ncrop\n%s", best_result['frame_level_report'])
        logger.info("classification video level report for ncrop\n%s", best_result['video_level_report'])
    return best_score


def train_one_step(
    args,
    nloader,
    aloader,
    model,
    optimizer,
    scheduler,
    scaler,
    criterion,
):
    """Training the model for one step."""
    model.train()
    optimizer.zero_grad()
    n_data_dict, a_data_dict = next(nloader), next(aloader)
    inputs = {k: torch.cat([n_data_dict[k], a_data_dict[k]], dim=0).to(args.device, non_blocking=True) for k in ["feats", "masks", "pseudo_label"]}
    with autocast(enabled=args.amp):
        scores, logits, masks, contrast_pairs = model(inputs, is_training=True)
    loss, loss_dict = criterion(scores, logits, masks, contrast_pairs, inputs["pseudo_label"])
    if args.amp:
        scale_before_step = scaler._scaler.get_scale()
        scaler(loss, optimizer, parameters=model.parameters(), clip_grad=1.0)
        skip_lr_sched = scale_before_step != scaler._scaler.get_scale()
        if not skip_lr_sched:
            scheduler.step()
    else:
        loss.backward()
        optimizer.step()
        scheduler.step()
    return loss_dict


def objective(trial, args, freeze_args):
    params = {
        # "learning_rate": trial.suggest_float("learning_rate", 1e-5, 8e-5, log=True),
        # "alpha": trial.suggest_float("alpha", 0, 0.5, step=0.1),
        # "gamma": trial.suggest_float("gamma", 1, 3.5, step=0.5),
        # "threshold": trial.suggest_float("threshold", 0.3, 0.5, step=0.05),
        # "lr_scheduler": trial.suggest_categorical("lr_scheduler", ["OneCycleLR", "constant"]),
        # "max_seq_len": trial.suggest_categorical("max_seq_len", [256, 384]),
        # "arch": trial.suggest_categorical("arch", ["1_2_5", "2_1_5"]),
        # "n_mha_win_size": trial.suggest_categorical("n_mha_win_size", [4, 8]),
        # "n_embd": trial.suggest_categorical("n_embd", [128, 256, 512, 1024]),
        # "n_head": trial.suggest_categorical("n_head", [4, 8]),
        # "se_ratio": trial.suggest_categorical("se_ratio", [4, 8]),
        # "dropout": trial.suggest_float("dropout", 0, 0.2, step=0.05),
    }
    # check if params are in freeze_args
    if freeze_args is not None:
        if any(k in freeze_args for k in params.keys()):
            raise ValueError(f"{set(freeze_args).intersection(set(params.keys()))} are in freeze_args and cannot be tuned! Please check")

    update_args_(args, params)
    score = train(args, trial)
    return score


if __name__ == "__main__":
    # Choose config params for different datasets
    for arg in sys.argv:
        if "ucfcrime" in str(arg):
            config_module = importlib.import_module("config.ucfcrime_cfg")
        elif "xdviolence" in str(arg):
            config_module = importlib.import_module("config.xdviolence_cfg")
    parser = config_module.parse_args()
    args = parser.parse_args()
    setup_device(args)
    setup_dataset(args)

    freeze_args = None
    if args.ckpt_file is not None:
        freeze_args = ["arch", "scale_factor", "max_seq_len", "n_mha_win_size", "n_embd", "n_head", "window_size", "se_ratio"]
        args_dict = torch.load(args.ckpt_file)["args"].__dict__
        freeze_params = {k: v for k, v in args_dict.items() if k in freeze_args}
        update_args_(args, freeze_params)

    if args.optuna:
        study = optuna.create_study(
            storage=f"sqlite:///{args.dataset_name}-db.sqlite3",
            pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=3),
            study_name=f"{args.dataset_name}-{time.strftime('%m%d_%H%M', time.localtime())}",
            direction="maximize",
        )
        study.optimize(lambda trial: objective(trial, args, freeze_args), n_trials=20)
        print(f"Best value: {study.best_value} (params: {study.best_params})")
    else:
        train(args)
