# main.py — MUKSB CLIP
"""
Entry point for CLIP machine unlearning experiments.

Usage
-----
  # Forget one class with MUKSB (KS bargaining):
  python main.py --unlearn MUKSB --dataset pets --arch ViT-B/32 \\
      --gpu 0 --save_dir results/muksb_clip

  # Compare against Nash baseline:
  python main.py --unlearn MUNBa --dataset pets --arch ViT-B/32 \\
      --gpu 0 --save_dir results/munba_clip
"""
import os
import sys
from collections import OrderedDict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import arg_parser
import torch
import torch.nn as nn
import utils

import clip
from unlearn import muksb, munba, FT, GA, SalUn, masked_nash


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

    # ── dataset ───────────────────────────────────────────────────────────────
    (
        train_loader_full,
        val_loader,
        test_loader,
        forget_loader,
        retain_loader,
        class_name,
    ) = utils.setup_dataset(args)
    retain_dataset = retain_loader.dataset
    forget_dataset = forget_loader.dataset

    print(f"Retain dataset size: {len(retain_dataset)}")
    print(f"Forget dataset size: {len(forget_dataset)}")

    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader,
        val=val_loader, test=test_loader,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model, preprocess = clip.load(args.arch, device=device)
    model.eval()

    prompts = [f"A photo of a {label}, a type of pet" for label in class_name]
    print(f"Prompts ({len(class_name)} classes):", prompts[:3], "...")
    texts = clip.tokenize(prompts).to(device)

    logit_scale = 100
    criterion   = nn.CrossEntropyLoss()
    evaluation_result = {}

    # ── unlearn ───────────────────────────────────────────────────────────────
    if args.unlearn == "MUKSB":
        muksb(texts, unlearn_data_loaders, model, args, class_name)
    elif args.unlearn == "MUNBa":
        munba(texts, unlearn_data_loaders, model, args, class_name)
    elif args.unlearn == "FT":
        FT.Finetune(texts, unlearn_data_loaders, model, args, class_name, with_l1=False)
    elif args.unlearn == "l1_sparse":
        FT.Finetune(texts, unlearn_data_loaders, model, args, class_name, with_l1=True)
    elif args.unlearn == "GA":
        GA.GradientAscent(texts, unlearn_data_loaders, model, args, class_name)
    elif args.unlearn == "SalUn":
        mask = torch.load(args.mask)
        SalUn.SaliencyUnlearn(texts, unlearn_data_loaders, model, args, class_name, mask=mask)
    elif args.unlearn == "masked_nash":
        masked_nash.MaskedNash(texts, unlearn_data_loaders, model, args, class_name)
    else:
        raise ValueError(f"Unknown unlearn method: {args.unlearn}")

    utils.save_checkpoint(model.state_dict(), False, args.save_dir, args.unlearn)

    # ── evaluate after unlearning ─────────────────────────────────────────────
    accuracy_unlearn = {}
    for name, loader in unlearn_data_loaders.items():
        print(name)
        utils.dataset_convert_to_test(loader.dataset, args)
        val_acc = utils.validate(loader, texts, logit_scale, model, criterion, device, args)
        accuracy_unlearn[name] = val_acc
        print(f"After unlearning, {name} acc: {val_acc}")
    evaluation_result["accuracy_unlearn"] = accuracy_unlearn
    utils.save_checkpoint(
        evaluation_result, False, args.save_dir, args.unlearn,
        filename="eval_result.pth.tar",
    )


if __name__ == "__main__":
    main()
