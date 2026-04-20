# https://huggingface.co/docs/diffusers/conceptual/evaluation
#compute-fid.py
import argparse

import torch
from dataset import setup_fid_data, setup_fid_data_i2p
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader


def compute_fid(class_to_forget, path, image_size):
    fid = FrechetInceptionDistance(feature=64)
    real_set, fake_set = setup_fid_data(class_to_forget, path, image_size)
    real_images = torch.stack(real_set).to(torch.uint8).cpu()
    fake_images = torch.stack(fake_set).to(torch.uint8).cpu()

    fid.update(real_images, real=True)  # doctest: +SKIP
    fid.update(fake_images, real=False)  # doctest: +SKIP
    fid_value = fid.compute()
    print(fid_value)
    fid.reset()
    del fid
    return fid_value.item() # doctest: +SKIP


# def compute_fid_i2p(real_path, path, image_size):
#     fid = FrechetInceptionDistance(feature=64)
#     real_set, fake_set = setup_fid_data_i2p(real_path, path, image_size, interpolation="bicubic")
#     real_images = torch.stack(real_set).to(torch.uint8).cpu()
#     fake_images = torch.stack(fake_set).to(torch.uint8).cpu()

#     fid.update(real_images, real=True)  # doctest: +SKIP
#     fid.update(fake_images, real=False)  # doctest: +SKIP
#     fid_value = fid.compute()
#     print(fid_value)
#     fid.reset()
#     del fid
#     return fid_value.item() # doctest: +SKIP


def compute_fid_i2p(real_path, path, image_size, batch_size=256):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fid = FrechetInceptionDistance(feature=2048).to(device)

    real_ds, fake_ds = setup_fid_data_i2p(
        real_path, path, image_size, interpolation="bicubic"
    )

    real_loader = DataLoader(real_ds, batch_size=batch_size, num_workers=4, pin_memory=True)
    fake_loader = DataLoader(fake_ds, batch_size=batch_size, num_workers=4, pin_memory=True)

    for batch in real_loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = ((batch + 1) * 127.5).clamp(0, 255).to(torch.uint8).to(device)
        fid.update(batch, real=True)

    for batch in fake_loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = ((batch + 1) * 127.5).clamp(0, 255).to(torch.uint8).to(device)
        fid.update(batch, real=False)

    score = fid.compute()
    print("FID:", score.item())
    fid.reset()
    return score.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generateImages", description="Generate Images using Diffusers Code"
    )
    parser.add_argument("--folder_path", help="path of images", type=str, required=False, default="/storage/s25017/MUKSB/SD/eval_scripts/Imagenette/cls9")
    parser.add_argument("--class_to_forget", type=int, nargs="+", required=False, default=[9],
                        help="One or more class indices to exclude (e.g. --class_to_forget 0 7)")
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument("--real_path", help="path of images", type=str, required=False,default="/storage/s25017/Datasets/COCO/coco_5k_val_2014_images")
    args = parser.parse_args()

    # image_size = args.image_size
    # path = args.folder_path
    # real_path = args.real_path
    # print(f"Computing FID...of {path} against {real_path}")
    # compute_fid_i2p(real_path, path, image_size)

    path = args.folder_path
    class_to_forget = args.class_to_forget
    image_size = args.image_size
    print(class_to_forget)
    compute_fid(class_to_forget, path, image_size)
