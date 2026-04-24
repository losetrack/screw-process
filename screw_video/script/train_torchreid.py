"""
Train an OSNet Re-ID model with torchreid on the prepared screw dataset.

Example:
    python script/train_torchreid.py
    python script/train_torchreid.py --use_cpu --workers 0
"""
import argparse
import os
import shutil
from pathlib import Path

import torch
import torchreid
from torchreid.reid.data import ImageDataManager
from torchreid.reid.engine import ImageSoftmaxEngine, ImageTripletEngine
from torchreid.reid.utils import load_pretrained_weights, set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train OSNet Re-ID with torchreid")
    parser.add_argument("--data_root", type=str, default="./reid_dataset_torchreid", help="Dataset root containing market1501")
    parser.add_argument("--dataset_name", type=str, default="market1501", help="Torchreid dataset name")
    parser.add_argument("--save_dir", type=str, default="./reid_runs/osnet_x0_25_screw", help="Training output directory")
    parser.add_argument("--export_path", type=str, default="./weights/osnet_x0_25_screw_reid.pth.tar", help="Export final checkpoint for inference")
    parser.add_argument("--model_name", type=str, default="osnet_x0_25", help="Re-ID backbone")
    parser.add_argument("--load_weights", type=str, default="./weights/osnet_x0_25_imagenet.pt", help="Initial pretrained weights")
    parser.add_argument("--loss", type=str, default="triplet", choices=["triplet", "softmax"], help="Training loss type")
    parser.add_argument("--gpu_devices", type=str, default=None, help="CUDA_VISIBLE_DEVICES value, e.g. '0'")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--height", type=int, default=256, help="Input image height")
    parser.add_argument("--width", type=int, default=128, help="Input image width")
    parser.add_argument("--transforms", nargs="*", default=["random_flip", "color_jitter", "random_erase"], help="Training transforms")
    parser.add_argument("--batch_size_train", type=int, default=32, help="Train batch size")
    parser.add_argument("--batch_size_test", type=int, default=64, help="Test batch size")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--num_instances", type=int, default=4, help="Images per identity when using RandomIdentitySampler")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--start_eval", type=int, default=10, help="Start evaluation after this epoch")
    parser.add_argument("--eval_freq", type=int, default=10, help="Evaluation frequency")
    parser.add_argument("--print_freq", type=int, default=10, help="Logging frequency")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--stepsize", type=int, default=20, help="LR step size")
    parser.add_argument("--gamma", type=float, default=0.1, help="LR decay factor")
    parser.add_argument("--margin", type=float, default=0.3, help="Triplet loss margin")
    parser.add_argument("--weight_t", type=float, default=1.0, help="Triplet loss weight")
    parser.add_argument("--weight_x", type=float, default=1.0, help="Softmax loss weight in triplet engine")
    return parser.parse_args()


def resolve_device(args):
    if args.gpu_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    use_gpu = torch.cuda.is_available() and not args.use_cpu
    return use_gpu


def build_datamanager(args, use_gpu):
    sampler = "RandomIdentitySampler" if args.loss == "triplet" else "RandomSampler"
    return ImageDataManager(
        root=args.data_root,
        sources=args.dataset_name,
        targets=args.dataset_name,
        height=args.height,
        width=args.width,
        transforms=args.transforms,
        batch_size_train=args.batch_size_train,
        batch_size_test=args.batch_size_test,
        workers=args.workers,
        num_instances=args.num_instances,
        train_sampler=sampler,
        use_gpu=use_gpu,
    )


def build_model(args, num_train_pids, use_gpu):
    model = torchreid.models.build_model(
        name=args.model_name,
        num_classes=num_train_pids,
        loss=args.loss,
        pretrained=False,
        use_gpu=use_gpu,
    )

    weights_path = Path(args.load_weights)
    if weights_path.exists():
        # The pretrained classifier head is expected to be skipped because the screw ID count differs.
        load_pretrained_weights(model, str(weights_path))
    else:
        print(f"[WARN] Initial weights not found, training from scratch: {weights_path}")

    if use_gpu:
        model = model.cuda()

    return model


def build_engine(args, datamanager, model, optimizer, scheduler, use_gpu):
    if args.loss == "triplet":
        return ImageTripletEngine(
            datamanager,
            model,
            optimizer=optimizer,
            margin=args.margin,
            weight_t=args.weight_t,
            weight_x=args.weight_x,
            scheduler=scheduler,
            use_gpu=use_gpu,
            label_smooth=True,
        )

    return ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        use_gpu=use_gpu,
        label_smooth=True,
    )


def export_final_checkpoint(save_dir, export_path):
    checkpoint_dir = Path(save_dir) / "model"
    checkpoint_files = sorted(checkpoint_dir.glob("model.pth.tar-*"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    latest_checkpoint = max(checkpoint_files, key=lambda path: int(path.name.split("-")[-1]))
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest_checkpoint, export_path)
    print(f"Exported final checkpoint to {export_path}")


def main():
    args = parse_args()
    set_random_seed(args.seed)
    use_gpu = resolve_device(args)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    datamanager = build_datamanager(args, use_gpu)
    model = build_model(args, datamanager.num_train_pids, use_gpu)

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim="adam",
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=args.stepsize,
        gamma=args.gamma,
        max_epoch=args.epochs,
    )

    engine = build_engine(args, datamanager, model, optimizer, scheduler, use_gpu)
    engine.run(
        save_dir=str(save_dir),
        max_epoch=args.epochs,
        print_freq=args.print_freq,
        start_eval=args.start_eval,
        eval_freq=args.eval_freq,
        dist_metric="cosine",
        normalize_feature=True,
    )

    if args.epochs > 0:
        export_final_checkpoint(save_dir, args.export_path)


if __name__ == "__main__":
    main()
