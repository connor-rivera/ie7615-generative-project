"""Comparison grids: one row per sweep caption, first column = the real COCO photo,
then one column per setting value. Only the column's setting differs across a row
(same caption, same seed), so every visible difference comes from that setting.

  python -m src.grids baseline
  python -m src.grids guidance | sampler | steps
"""
import argparse
import json
import textwrap

from PIL import Image, ImageDraw, ImageFont

from src.data import MANIFEST_DIR, load_image, load_manifest
from src.sweep import IMG_DIR, OUT_DIR, conditions, image_name

THUMB = 256
LABEL_W = 260
HEADER_H = 40


def _font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


def make_grid(sweep):
    prompts = json.load(open(MANIFEST_DIR / "sweep_prompts.json"))["prompts"]
    by_id = {e["image_id"]: e for e in load_manifest()["images"]}
    conds = conditions(sweep)
    col_labels = ["real COCO photo"] + [
        "baseline (DDIM, 30, 7.5)" if knob == "baseline" else f"{knob} = {value}"
        for knob, value, _ in conds]

    W = LABEL_W + THUMB * len(col_labels)
    H = HEADER_H + THUMB * len(prompts)
    grid = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(grid)
    f_head, f_cap = _font(18), _font(15)

    for c, label in enumerate(col_labels):
        draw.text((LABEL_W + c * THUMB + 8, 10), label, fill="black", font=f_head)

    for r, p in enumerate(prompts):
        y = HEADER_H + r * THUMB
        text = f"[{p['type']}] {p['prompt']}"
        draw.multiline_text((8, y + 8), "\n".join(textwrap.wrap(text, 30)),
                            fill="black", font=f_cap, spacing=3)
        real = load_image(by_id[p["image_id"]]).resize((THUMB, THUMB), Image.BICUBIC)
        grid.paste(real, (LABEL_W, y))
        for c, (_, _, s) in enumerate(conds, start=1):
            path = IMG_DIR / image_name(s, p["image_id"], p["seed"])
            if path.exists():
                grid.paste(Image.open(path).convert("RGB").resize((THUMB, THUMB), Image.BICUBIC),
                           (LABEL_W + c * THUMB, y))
            else:
                draw.text((LABEL_W + c * THUMB + 8, y + 8), "(not generated)", fill="gray",
                          font=f_cap)

    out = OUT_DIR / "grids" / f"grid_{sweep}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out, quality=90)
    print(f"Saved {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", choices=["baseline", "guidance", "sampler", "steps"])
    make_grid(parser.parse_args().sweep)


if __name__ == "__main__":
    main()
