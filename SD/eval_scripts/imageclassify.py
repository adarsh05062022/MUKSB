import argparse
import os

import pandas as pd
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ImageClassification",
        description="Takes the path of images and generates classification scores",
    )

    parser.add_argument(
        "--prompts_path",
        help="path to prompts",
        type=str,
        required=False,
        default="/storage/s25017/MUKSB/SD/eval_scripts/CLASS/prompts.csv",
    )

    parser.add_argument(
        "--save_path",
        help="base path to save results",
        type=str,
        required=False,
        default="CLASS/UA/multiclass",
    )

    parser.add_argument("--device", type=str, required=False, default="cuda:0")
    parser.add_argument("--topk", type=int, required=False, default=5)
    parser.add_argument("--batch_size", type=int, required=False, default=250)

    args = parser.parse_args()

    device = args.device
    topk = args.topk
    batch_size = args.batch_size
    prompts_path = args.prompts_path
    save_path = args.save_path

    # -------- LIST OF IMAGE FOLDERS --------
    folders = [
      "/storage/s25017/MUKSB/SD/eval_scripts/CLASS/gen/diffusers-cls_0_4-MUNBa-method_full-lr_1e-05_E8_U1904_multi_topk10_an_image.pt",
      "/storage/s25017/MUKSB/SD/eval_scripts/CLASS/gen/diffusers-cls_0_7_4-MUNBa-method_full-lr_1e-05_E8_U2835_multi_topk10_an_image.pt",
      "/storage/s25017/MUKSB/SD/eval_scripts/CLASS/gen/diffusers-cls_0_7-MUNBa-method_full-lr_1e-05_E8_U1894_multi_topk10_an_image.pt",
      "/storage/s25017/MUKSB/SD/eval_scripts/CLASS/gen/diffusers-cls_7_4-MUNBa-method_full-lr_1e-05_E8_U1872_multi_topk10_an_image.pt"


    ]

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.to(device)
    model.eval()

    preprocess = weights.transforms()

    for folder in folders:

        print(f"Processing folder: {folder}")

        scores = {}
        categories = {}
        indexes = {}

        for k in range(1, topk + 1):
            scores[f"top{k}"] = []
            indexes[f"top{k}"] = []
            categories[f"top{k}"] = []

        names = os.listdir(folder)
        names = [name for name in names if ".png" in name or ".jpg" in name]

        images = []
        for name in names:
            img = Image.open(os.path.join(folder, name))
            batch = preprocess(img)
            images.append(batch)

        if batch_size is None:
            batch_size = len(names)

        if batch_size > len(names):
            batch_size = len(names)

        images = torch.stack(images)

        for i in range(((len(names) - 1) // batch_size) + 1):

            batch = images[i * batch_size : min(len(names), (i + 1) * batch_size)].to(device)

            with torch.no_grad():
                prediction = model(batch).softmax(1)

            probs, class_ids = torch.topk(prediction, topk, dim=1)

            for k in range(1, topk + 1):
                scores[f"top{k}"].extend(probs[:, k - 1].detach().cpu().numpy())
                indexes[f"top{k}"].extend(class_ids[:, k - 1].detach().cpu().numpy())

                categories[f"top{k}"].extend(
                    [
                        weights.meta["categories"][idx]
                        for idx in class_ids[:, k - 1].detach().cpu().numpy()
                    ]
                )

        df = pd.read_csv(prompts_path)
        df["case_number"] = df["case_number"].astype("int")

        case_numbers = []
        for name in names:
            case_number = (
                name.split("_")[0]
                .replace(".png", "")
                .replace(".jpg", "")
            )
            case_numbers.append(int(case_number))

        dict_final = {"case_number": case_numbers}

        for k in range(1, topk + 1):
            dict_final[f"category_top{k}"] = categories[f"top{k}"]
            dict_final[f"index_top{k}"] = indexes[f"top{k}"]
            dict_final[f"scores_top{k}"] = scores[f"top{k}"]

        df_results = pd.DataFrame(dict_final)
        merged_df = pd.merge(df, df_results)

        folder_name = folder.split("/")[-1]
        output_csv = os.path.join(save_path, f"{folder_name}_classification.csv")

        merged_df.to_csv(output_csv, index=False)

        print(f"Saved results to {output_csv}")