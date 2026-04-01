"""
CLIP/generate_mask.py — MUKSB
Gradient-magnitude saliency mask generation for CLIP unlearning.

Usage
-----
  python generate_mask.py --unlearn MUKSB --dataset pets --arch ViT-B/32 \\
      --mode all --gpu 0 --save_dir masks/clip_pets

Saves masks/clip_pets/with_0.5.pt — binary dict (param_name → 0/1 tensor).
"""
import copy
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

    # Dataset
    (train_loader_full, val_loader, test_loader, forget_loader, retain_loader, class_name
     ) = utils.setup_dataset(args)

    # Model
    model, preprocess = clip.load(args.arch, device=device)
    model.eval()

    prompts = [f"A photo of a {label}, a type of pet" for label in class_name]
    texts   = clip.tokenize(prompts).to(device)
    logit_scale = 100
    criterion = nn.CrossEntropyLoss()

    # Freeze all, selectively unfreeze
    gradients = {}
    for param in model.parameters():
        param.requires_grad = False

    if args.mode == "text":
        print("Unfreezing text encoder")
        for name, param in model.transformer.named_parameters():
            param.requires_grad = True; gradients[name] = 0
        optimizer = torch.optim.SGD(model.transformer.parameters(), args.unlearn_lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.mode == "image":
        print("Unfreezing visual encoder")
        for name, param in model.visual.named_parameters():
            param.requires_grad = True; gradients[name] = 0
        optimizer = torch.optim.SGD(model.visual.parameters(), args.unlearn_lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    else:  # "all"
        print("Unfreezing all parameters")
        for name, param in model.named_parameters():
            param.requires_grad = True; gradients[name] = 0
        optimizer = torch.optim.SGD(model.parameters(), args.unlearn_lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)

    # Accumulate |∇L_forget| (gradient ascent direction)
    for image, target in forget_loader:
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad()
        if args.mode == "text":
            with torch.no_grad(): image_features = model.encode_image(image)
            text_features = model.encode_text(texts)
        elif args.mode == "image":
            image_features = model.encode_image(image)
            with torch.no_grad(): text_features = model.encode_text(texts)
        else:
            image_features = model.encode_image(image)
            text_features  = model.encode_text(texts)
        text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        cos_sim = logit_scale * image_features @ text_features.t()
        loss = -criterion(cos_sim, target)
        loss.backward()
        with torch.no_grad():
            src = (model.transformer if args.mode == "text" else
                   model.visual if args.mode == "image" else model)
            for name, param in src.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

    # Threshold at 50% density (top-50% most salient parameters)
    threshold_list = [0.5]
    for density in threshold_list:
        all_elements  = -torch.cat([t.flatten() for t in gradients.values()])
        threshold_idx = int(len(all_elements) * density)
        positions = torch.argsort(all_elements)
        ranks     = torch.argsort(positions)
        hard_dict = {}
        start = 0
        for key, tensor in gradients.items():
            n = tensor.numel()
            tensor_ranks = ranks[start: start + n]
            mask = torch.zeros_like(tensor_ranks)
            mask[tensor_ranks < threshold_idx] = 1
            hard_dict[key] = mask.reshape(tensor.shape)
            start += n
        save_path = os.path.join(args.save_dir, f"with_{density}.pt")
        torch.save(hard_dict, save_path)
        print(f"Saved mask (density={density}) → {save_path}")


if __name__ == "__main__":
    main()
