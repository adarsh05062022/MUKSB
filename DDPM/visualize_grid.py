"""
Creates:
  1. A 10x10 combined canvas
  2. 10 individual row canvases (one per unlearned class)

Layout:
  - Rows = Unlearned Class (which class was forgotten)
  - Cols = Original Class  (which class was used to generate)
  - Diagonal highlight = unlearn class == prompt class
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

CIFAR10_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

BASE_DIR = Path("/scratch/s25017/MUKSB/DDPM/results/cifar10/forget/rl/0.001_no_mask")

IMG_SIZE    = 64   # upsample 32→64 for visibility
CELL_PAD    = 2    # tighter gap between cells
CELL_SIZE   = IMG_SIZE + 2 * CELL_PAD
LABEL_W     = 80
LABEL_H     = 80
DIAG_BORDER = 3
BG_COLOR    = (255, 255, 255)
DIAG_COLOR  = (148, 103, 189)
FONT_SIZE   = 10
TITLE_FONT_SIZE = 11
AXIS_TITLE_W = 18
TITLE_H      = 24


def load_fonts():
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    reg_path  = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        bold = ImageFont.truetype(bold_path, TITLE_FONT_SIZE)
        reg  = ImageFont.truetype(bold_path, FONT_SIZE)   # bold for class names too
    except Exception:
        bold = reg = ImageFont.load_default()
    return bold, reg


def get_latest_run(class_dir: Path):
    if not class_dir.exists():
        return None
    runs = sorted(class_dir.iterdir(), reverse=True)
    return next((r for r in runs if r.is_dir()), None)


def pick_image(forget_dir, prompt_class: int, index: int = 0):
    if forget_dir is None:
        return None
    img_dir = forget_dir / f"class{prompt_class}_images"
    if not img_dir.exists():
        return None
    imgs = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    return imgs[index] if index < len(imgs) else None


def load_or_blank(path) -> Image.Image:
    if path and path.exists():
        img = Image.open(path).convert("RGB")
        return img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    blank = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (210, 210, 210))
    d = ImageDraw.Draw(blank)
    d.line([(0, 0), (IMG_SIZE, IMG_SIZE)], fill=(180, 180, 180), width=2)
    d.line([(IMG_SIZE, 0), (0, IMG_SIZE)], fill=(180, 180, 180), width=2)
    return blank


def centered_text(draw, text, font, x, y, w, h, color=(30, 30, 30)):
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.text((x + (w - tw) // 2, y + (h - th) // 2), text, fill=color, font=font)


def make_rotated_label(text: str, font, cell_w: int, label_h: int) -> Image.Image:
    """Horizontal label rotated 90° CCW so text reads bottom-to-top."""
    bb = font.getbbox(text)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    # horizontal surface: width=label_h, height=cell_w
    surf = Image.new("RGB", (label_h, cell_w), BG_COLOR)
    d = ImageDraw.Draw(surf)
    d.text(((label_h - tw) // 2, (cell_w - th) // 2), text, fill=(30, 30, 30), font=font)
    return surf.rotate(90, expand=True)   # now (cell_w wide, label_h tall)


def draw_axis_title_vertical(canvas: Image.Image, text: str, font, strip_w: int,
                              grid_top: int, grid_h: int):
    draw = ImageDraw.Draw(canvas)
    bb = draw.textbbox((0, 0), text, font=font)
    tw = bb[2] - bb[0]
    surf = Image.new("RGB", (tw + 4, strip_w), BG_COLOR)
    sd = ImageDraw.Draw(surf)
    sd.text((2, (strip_w - (bb[3] - bb[1])) // 2), text, fill=(30, 30, 30), font=font)
    rotated = surf.rotate(90, expand=True)
    y = grid_top + (grid_h - rotated.height) // 2
    canvas.paste(rotated, (0, y))


def build_canvas(rows, bold_font, label_font, n_cols=10, img_index=0):
    """
    Build a canvas for the given list of (unlearn_class_idx, forget_dir) rows.
    rows: list of (row_idx, forget_dir_or_None)
    """
    n_rows = len(rows)
    grid_w  = LABEL_W + n_cols * CELL_SIZE
    grid_h  = LABEL_H + n_rows * CELL_SIZE
    total_w = AXIS_TITLE_W + grid_w
    total_h = TITLE_H + grid_h

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)

    # ── top title "Original Class" ──────────────────────────────────────────
    top_title = "Original Class"
    centered_text(draw, top_title, bold_font,
                  x=AXIS_TITLE_W + LABEL_W, y=0,
                  w=n_cols * CELL_SIZE, h=TITLE_H)

    # ── left axis title "Unlearned Class" ───────────────────────────────────
    draw_axis_title_vertical(canvas, "Unlearned Class", bold_font,
                             strip_w=AXIS_TITLE_W,
                             grid_top=TITLE_H + LABEL_H,
                             grid_h=n_rows * CELL_SIZE)

    # ── col labels (rotated) ─────────────────────────────────────────────────
    for col in range(n_cols):
        lbl = make_rotated_label(CIFAR10_CLASSES[col], label_font,
                                 cell_w=CELL_SIZE, label_h=LABEL_H - 4)
        x = AXIS_TITLE_W + LABEL_W + col * CELL_SIZE + (CELL_SIZE - lbl.width) // 2
        y = TITLE_H + (LABEL_H - lbl.height) // 2
        canvas.paste(lbl, (x, y))

    # ── row labels ────────────────────────────────────────────────────────────
    for i, (row_idx, _) in enumerate(rows):
        y0 = TITLE_H + LABEL_H + i * CELL_SIZE
        centered_text(draw, CIFAR10_CLASSES[row_idx], label_font,
                      x=AXIS_TITLE_W, y=y0,
                      w=LABEL_W, h=CELL_SIZE)

    # ── grid cells ────────────────────────────────────────────────────────────
    for i, (row_idx, forget_dir) in enumerate(rows):
        for col in range(n_cols):
            img_path = pick_image(forget_dir, col, index=img_index)
            img      = load_or_blank(img_path)

            x0 = AXIS_TITLE_W + LABEL_W + col * CELL_SIZE
            y0 = TITLE_H + LABEL_H + i * CELL_SIZE

            canvas.paste(img, (x0 + CELL_PAD, y0 + CELL_PAD))

            if row_idx == col:   # diagonal
                for t in range(DIAG_BORDER):
                    draw.rectangle(
                        [x0 + t, y0 + t,
                         x0 + CELL_SIZE - 1 - t,
                         y0 + CELL_SIZE - 1 - t],
                        outline=DIAG_COLOR
                    )

    return canvas


def main():
    bold_font, label_font = load_fonts()

    # Gather all row data
    all_rows = []
    for cls in range(10):
        class_dir  = BASE_DIR / f"class{cls}"
        run_dir    = get_latest_run(class_dir)
        forget_dir = run_dir / f"class{cls}_forget" if run_dir else None
        all_rows.append((cls, forget_dir))

    # ── 10 canvases: canvas N uses image N.png from every cell ───────────────
    for idx in range(10):
        canvas = build_canvas(all_rows, bold_font, label_font, img_index=idx)
        out = BASE_DIR / f"grid_canvas_{idx}.png"
        canvas.save(out, dpi=(150, 150))
        print(f"Saved canvas {idx} → {out}")


if __name__ == "__main__":
    main()
