# main_forget.py — MUKSB Classification
"""
Entry point for machine unlearning experiments on classification models.

Usage
-----
  # Forget class 1 with MUKSB (KS bargaining):
  python main_forget.py --unlearn MUKSB --class_to_replace 1 \\
      --dataset cifar10 --arch resnet18 --gpu 0 \\
      --mask <path/to/pretrained.pth> --save_dir results/muksb_cls1

  # Compare against Nash baseline:
  python main_forget.py --unlearn MUNBa --class_to_replace 1 \\
      --dataset cifar10 --arch resnet18 --gpu 0 \\
      --mask <path/to/pretrained.pth> --save_dir results/munba_cls1
"""
import copy
import os
import sys
from collections import OrderedDict

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import arg_parser
import evaluation
import torch
import torch.nn as nn
import unlearn          # MUKSB/Classification/unlearn (our package)
import utils

from trainer import validate
from evaluation import mia_


def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed

    # ── dataset / model setup ────────────────────────────────────────────────
    if args.dataset != "celeba":
        (
            model,
            train_loader_full,
            val_loader,
            test_loader,
            marked_loader,
        ) = utils.setup_model_dataset(args)
        model.cuda()

        def replace_loader_dataset(dataset, batch_size=args.batch_size, seed=1, shuffle=True):
            utils.setup_seed(seed)
            return torch.utils.data.DataLoader(
                dataset, batch_size=batch_size,
                num_workers=0, pin_memory=True, shuffle=shuffle,
            )

        forget_dataset = copy.deepcopy(marked_loader.dataset)
        if args.dataset == "svhn":
            try:
                marked = forget_dataset.targets < 0
            except AttributeError:
                marked = forget_dataset.labels < 0
            forget_dataset.data = forget_dataset.data[marked]
            try:
                forget_dataset.targets = -forget_dataset.targets[marked] - 1
            except AttributeError:
                forget_dataset.labels = -forget_dataset.labels[marked] - 1
            forget_loader  = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            try:
                marked = retain_dataset.targets >= 0
            except AttributeError:
                marked = retain_dataset.labels >= 0
            retain_dataset.data = retain_dataset.data[marked]
            try:
                retain_dataset.targets = retain_dataset.targets[marked]
            except AttributeError:
                retain_dataset.labels = retain_dataset.labels[marked]
            retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        else:
            try:
                marked = forget_dataset.targets < 0
                forget_dataset.data    = forget_dataset.data[marked]
                forget_dataset.targets = -forget_dataset.targets[marked] - 1
                forget_loader  = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
                retain_dataset = copy.deepcopy(marked_loader.dataset)
                marked = retain_dataset.targets >= 0
                retain_dataset.data    = retain_dataset.data[marked]
                retain_dataset.targets = retain_dataset.targets[marked]
                retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
            except Exception:
                marked = forget_dataset.targets < 0
                forget_dataset.imgs    = forget_dataset.imgs[marked]
                forget_dataset.targets = -forget_dataset.targets[marked] - 1
                forget_loader  = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
                retain_dataset = copy.deepcopy(marked_loader.dataset)
                marked = retain_dataset.targets >= 0
                retain_dataset.imgs    = retain_dataset.imgs[marked]
                retain_dataset.targets = retain_dataset.targets[marked]
                retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)

        assert len(forget_dataset) + len(retain_dataset) == len(train_loader_full.dataset)
    else:
        (
            model,
            train_loader_full,
            val_loader,
            test_loader,
            forget_loader,
            retain_loader,
        ) = utils.setup_model_dataset(args)
        model.cuda()
        retain_dataset = retain_loader.dataset
        forget_dataset = forget_loader.dataset

    print(f"Retain dataset size: {len(retain_dataset)}")
    print(f"Forget dataset size: {len(forget_dataset)}")

    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader,
        val=val_loader, test=test_loader,
    )
    criterion = nn.CrossEntropyLoss()
    evaluation_result = None

    # ── load checkpoint ──────────────────────────────────────────────────────
    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        if args.unlearn not in ("retrain", "raw"):
            ckpt = torch.load(args.mask, map_location=device)
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            model.load_state_dict(ckpt, strict=False)

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        unlearn_method(unlearn_data_loaders, model, criterion, args)
        unlearn.save_unlearn_checkpoint(model, None, args)

    # ── evaluation ───────────────────────────────────────────────────────────
    if evaluation_result is None:
        evaluation_result = {}

    if "new_accuracy" not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            print(name)
            utils.dataset_convert_to_test(loader.dataset, args)
            val_acc = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")
        evaluation_result["accuracy"] = accuracy
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        evaluation_result.pop(deprecated, None)

    if "MIA" not in evaluation_result:
        utils.dataset_convert_to_test(retain_loader.dataset, args)
        utils.dataset_convert_to_test(forget_loader.dataset, args)
        utils.dataset_convert_to_test(test_loader.dataset,   args)

        mia = mia_.get_mia(model, forget_loader, test_loader, device)
        print(f"MIA accuracy on forgotten vs unseen: {mia:.3f}")
        evaluation_result["MIA"] = mia

    if "SVC_MIA_forget_efficacy" not in evaluation_result:
        test_len = len(test_loader.dataset)
        utils.dataset_convert_to_test(retain_dataset,    args)
        utils.dataset_convert_to_test(forget_loader,     args)
        utils.dataset_convert_to_test(test_loader,       args)

        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )
        evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=test_loader,
            target_train=None,
            target_test=forget_loader,
            model=model,
        )
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    if "SVC_MIA_training_privacy" not in evaluation_result:
        test_len   = len(test_loader.dataset)
        retain_len = len(retain_dataset)
        num = test_len // 2
        utils.dataset_convert_to_test(retain_dataset, args)
        utils.dataset_convert_to_test(forget_loader,  args)
        utils.dataset_convert_to_test(test_loader,    args)

        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(num)))
        target_train = torch.utils.data.Subset(retain_dataset, list(range(num, retain_len)))
        shadow_test  = torch.utils.data.Subset(test_loader.dataset, list(range(num)))
        target_test  = torch.utils.data.Subset(test_loader.dataset, list(range(num, test_len)))

        evaluation_result["SVC_MIA_training_privacy"] = evaluation.SVC_MIA(
            shadow_train=torch.utils.data.DataLoader(shadow_train, batch_size=args.batch_size, shuffle=False),
            shadow_test=torch.utils.data.DataLoader(shadow_test,   batch_size=args.batch_size, shuffle=False),
            target_train=torch.utils.data.DataLoader(target_train, batch_size=args.batch_size, shuffle=False),
            target_test=torch.utils.data.DataLoader(target_test,   batch_size=args.batch_size, shuffle=False),
            model=model,
        )
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == "__main__":
    main()
