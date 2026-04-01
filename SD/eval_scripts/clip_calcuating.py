import os
import torch
import pandas as pd
from PIL import Image,ImageFile
import clip
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = False 

# -----------------------------
# Paths
# -----------------------------
image_folder = "/storage/s25017/MUKSB/SD/eval_scripts/CLASS/COCO30k_NSFW_beta_100_e14"
prompt_csv = "/storage/s25017/Datasets/COCO/coco_30k.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load CLIP model
# -----------------------------
model, preprocess = clip.load("ViT-B/32", device=device)

# -----------------------------
# Load prompt file
# -----------------------------
df = pd.read_csv(prompt_csv)

scores = []
used_samples = 0

# -----------------------------
# Iterate over prompts
# -----------------------------
for _, row in tqdm(df.iterrows(), total=len(df)):

    case_number = row["case_number"]
    prompt = row["prompt"]

    image_name = f"{case_number}_0.png"
    image_path = os.path.join(image_folder, image_name)

    # Skip if image does not exist
    if not os.path.exists(image_path):
        continue

    try:
        img = Image.open(image_path).convert("RGB")
        image = preprocess(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"[CORRUPTED] {image_name}")
        continue
    text = clip.tokenize([prompt]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).item()

    scores.append(similarity)
    used_samples += 1

# -----------------------------
# Final score
# -----------------------------
clip_score = sum(scores) / len(scores)

print("Images used:", used_samples)
print("CLIP Score:", clip_score * 100)