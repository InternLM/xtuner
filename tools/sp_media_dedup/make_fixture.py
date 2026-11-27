#!/usr/bin/env python3
"""B0 synthetic fixture: tiny local images + a mini annotation in the target format.

Covers: multi-image row, byte-identical duplicate media under two paths (content-hash
vs path keying), text-only row (pure_text_get_item), and distinct-image row.
Sizes are small so upscaled patch counts stay tiny (min_image_tokens floor applies).
"""
import json
import os
import sys


def make_image(path, w, h, rgb):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (w, h), rgb)
    d = ImageDraw.Draw(img)
    for i in range(0, w, 8):
        d.line([(i, 0), (i, h)], fill=(255 - rgb[0], rgb[1], rgb[2]), width=1)
    d.rectangle([w // 4, h // 4, 3 * w // 4, 3 * h // 4], fill=rgb)
    img.save(path, format="PNG")


def main(out_dir):
    media = os.path.join(out_dir, "media")
    os.makedirs(media, exist_ok=True)
    specs = {
        "img_0.png": (56, 56, (200, 30, 30)),
        "img_1.png": (112, 84, (30, 200, 30)),
        "img_2.png": (84, 112, (30, 30, 200)),
        "img_3.png": (140, 56, (200, 200, 30)),
    }
    for name, (w, h, rgb) in specs.items():
        make_image(os.path.join(media, name), w, h, rgb)
    # byte-identical duplicate under a different path
    with open(os.path.join(media, "img_0.png"), "rb") as f:
        dup = f.read()
    with open(os.path.join(media, "img_0_dup.png"), "wb") as f:
        f.write(dup)

    def img(url, wh):
        # mirror the real target annotation format: image entries carry image_wh metadata
        # (required by the cache-path geometry prediction; rows without it tokenize to
        # num_tokens=0 and get filtered as damaged)
        return {"type": "image", "image": {"url": url, "image_wh": list(wh)}}

    rows = [
        {"id": "fix-0000", "messages": [
            {"role": "user", "content": [img("img_0.png", (56, 56)), img("img_0_dup.png", (56, 56)),
                                          {"type": "text", "text": "Two identical chart images: describe them."}]},
            {"role": "assistant", "content": "Both images show the same striped chart pattern."}]},
        {"id": "fix-0001", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "What is 2+2? Answer briefly."}]},
            {"role": "assistant", "content": "2+2 equals 4."}]},
        {"id": "fix-0002", "messages": [
            {"role": "user", "content": [img("img_1.png", (112, 84)),
                                         {"type": "text", "text": "Describe this green figure."}]},
            {"role": "assistant", "content": "A green block with vertical stripes."}]},
        {"id": "fix-0003", "messages": [
            {"role": "user", "content": [img("img_1.png", (112, 84)), img("img_2.png", (84, 112)),
                                         img("img_3.png", (140, 56)),
                                         {"type": "text", "text": "Compare these three figures."}]},
            {"role": "assistant", "content": "They differ in color and aspect ratio."}]},
    ]
    ann = os.path.join(out_dir, "ann.jsonl")
    with open(ann, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(ann)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/b0")
