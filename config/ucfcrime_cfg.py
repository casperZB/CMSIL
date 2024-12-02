import argparse

# fmt: off
def parse_args():
    parser = argparse.ArgumentParser(description="Video Anomaly Transformer")

    # ========================= Train Configs ==========================
    parser.add_argument("--seed", type=int, default=3407, help="random seed.")
    parser.add_argument("--batch_size", default=30, type=int, help="number of instances in a batch of data")
    parser.add_argument("--test_batch_size", default=10, type=int, help="test batch size")
    parser.add_argument("--modality", type=str, default="rgb", help="the modality of the input: [rgb, flow, rgb_flow]")
    parser.add_argument("--max_step", type=int, default=2000, help="maximum iteration to train")
    parser.add_argument("--ckpt_file", type=str, default=None, help="saved model")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=1.0307560117555227e-5, type=float, help="initial learning rate")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--save_score", default=80, type=float, help="save checkpoint if best metric score > save_score")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AUTOMATIC MIXED PRECISION")
    parser.add_argument("--lr_scheduler", type=str, default="OneCycleLR", help="type of lr scheduler: [OneCycleLR, cosine, constant]")
    parser.add_argument("--eval_metric", type=str, default="AUC@ROC", help="the evalate metric: [AUC@PR, AUC@ROC, ACC]")
    parser.add_argument("--optuna", action="store_true", help="Enable optuna hyperparameter search")
    parser.add_argument("--alpha", default=0.2, type=float, help="alpha for focal loss")
    parser.add_argument("--gamma", default=1.0, type=float, help="gamma for focal loss, if gamma=0, then use BCELoss")
    parser.add_argument("--threshold", default=0.45, type=float, help="threshold for pseudo label")

    # ========================= Model Configs ==========================
    parser.add_argument("--arch", type=lambda s: [int(i) for i in s.split('_')], default="1_2_5", help="[convs, stem transformers, branch transformers]")
    parser.add_argument("--scale_factor", type=int, default=2, help="scale factor between fpn branch layers.")
    parser.add_argument("--max_seq_len", type=int, default=128, help="max sequence length (used for training)")
    parser.add_argument("--n_mha_win_size", type=int, default=8, help="window size for self attention; -1 to use full seq")
    parser.add_argument("--n_embd", type=int, default=1024, help="embedding dimension")
    parser.add_argument("--n_head", type=int, default=4, help="number of head for self-attention in transformers")
    parser.add_argument("--window_size", type=int, default=16, help="The number of frames from which to extract features (or window size")
    parser.add_argument("--se_ratio", type=int, default=4, help="reduction factor in se context gating")
    parser.add_argument("--dropout", type=float, default=0, help="dropout ratio")

    # ========================= Data Configs ==========================
    parser.add_argument("--zip_feats", type=str, required=True, help="dataset zip path")
    parser.add_argument('--pseudo_label', type=str, default='data/UCF_frame_pseudo_label.npy', help="the pseudo label path")
    parser.add_argument("--num_workers", default=4, type=int, help="num_workers for dataloaders")
    parser.add_argument("--tencrop", action="store_false", help="ten-crop data augmentation in testing stage")

    return parser
