#!/usr/bin/env python3
"""Watermark images with a visible mark plus a hidden ID in the pixels.

    python watermark.py embed  <images or folders> --out <folder> --owner "Your Name"
    python watermark.py check  <suspect image> [...]

embed  Writes watermarked COPIES into --out. It never writes into a folder that
       holds an input image, so originals stay untouched. Each copy gets:
         1. a tiled visible text mark (skip with --no-visible),
         2. a random 64-bit ID hidden in the pixels (DWT-DCT-SVD, blind-watermark),
            after squeezing brightness by --headroom levels at each end so pure
            white and pure black have room to carry it (white 255 becomes 243),
         3. a copyright note in the file metadata (trivially strippable).
       Every ID is recorded in manifest.json beside this script.

check  Reads the hidden ID out of a suspect image and compares it with every ID
       in manifest.json. It tries the image as-is and mirrored, then with a
       brightness/contrast edit undone, then re-aligned against each published
       copy to undo cropping. The last two need the published copies to still be
       at the paths recorded in manifest.json. It reports how
       many of the 64 bits agree and the odds of that happening by chance.

Files that must be kept private and backed up. They live beside this script,
or in the folder named by the WATERMARK_DIR environment variable:
    watermark.key   secret embedding key. Without it no ID can be read back.
    manifest.json   which ID went into which image.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import secrets
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import cv2
import blind_watermark
from blind_watermark import WaterMark

if hasattr(blind_watermark, "bw_notes"):
    blind_watermark.bw_notes.close()

HERE = Path(os.environ.get("WATERMARK_DIR") or Path(__file__).resolve().parent).resolve()
KEY_FILE = HERE / "watermark.key"
MANIFEST = HERE / "manifest.json"

ID_BITS = 64
MATCH_BITS = 58      # >= this many of 64 agreeing: treated as a match
LIKELY_BITS = 52     # >= this many: likely, reported with its odds
MIN_SIDE = 256       # smaller images do not hold the ID reliably
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",       # Debian/Ubuntu
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",     # Fedora
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",                   # Arch
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",          # macOS
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",                               # Windows
    "C:/Windows/Fonts/segoeuib.ttf",
)


# ----------------------------------------------------------------- key / manifest
def load_key(create):
    """Return (password_img, password_wm). The file is parsed, never executed."""
    if not KEY_FILE.exists():
        if not create:
            sys.exit(f"No key file at {KEY_FILE}. Nothing has been watermarked with this tool yet.")
        fd = os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(f"password_img={secrets.randbelow(2**31 - 2) + 1}\n")
            f.write(f"password_wm={secrets.randbelow(2**31 - 2) + 1}\n")
        print(f"Created new secret key: {KEY_FILE}  (back this file up)")
    values = {}
    for line in KEY_FILE.read_text().splitlines():
        line = line.rstrip("\r")
        for name in ("password_img", "password_wm"):
            if line.startswith(name + "="):
                values[name] = int(line[len(name) + 1:])
    if set(values) != {"password_img", "password_wm"}:
        sys.exit(f"{KEY_FILE} is missing password_img or password_wm.")
    return values["password_img"], values["password_wm"]


def load_manifest():
    if not MANIFEST.exists():
        return []
    return json.loads(MANIFEST.read_text())


def save_manifest(entries):
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries, indent=2) + "\n")
    os.replace(tmp, MANIFEST)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------- image helpers
def open_rgb(path):
    """Return (RGB image, alpha channel or None), with EXIF rotation applied."""
    im = ImageOps.exif_transpose(Image.open(path))
    alpha = None
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        alpha = im.getchannel("A")
    return im.convert("RGB"), alpha


def to_bgr(im):
    return np.ascontiguousarray(np.asarray(im, dtype=np.uint8)[:, :, ::-1])


def from_bgr(arr):
    arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(arr[:, :, ::-1]), "RGB")


def find_font(size, font_path=None):
    for path in ((font_path,) if font_path else FONT_CANDIDATES):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    if font_path:
        sys.exit(f"Font not found: {font_path}")
    try:
        return ImageFont.load_default(size)   # Pillow >= 10.1 ships a scalable default
    except TypeError:
        sys.exit("No usable font found for the visible mark; pass one with --font.")


def add_visible(im, text, opacity, font_path=None):
    """Tile semi-transparent diagonal text over the whole image."""
    w, h = im.size
    font = find_font(max(14, round(min(w, h) / 22)), font_path)
    side = int(math.hypot(w, h)) + 4
    layer = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    box = draw.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    step_x, step_y = int(tw * 1.45), int(th * 5.0)
    a = round(255 * opacity)
    for row, y in enumerate(range(0, side, step_y)):
        for x in range(-step_x if row % 2 else -step_x // 2, side, step_x):
            draw.text((x, y), text, font=font, fill=(255, 255, 255, a),
                      stroke_width=max(1, font.size // 22), stroke_fill=(0, 0, 0, a))
    layer = layer.rotate(30, resample=Image.BICUBIC)
    left, top = (side - w) // 2, (side - h) // 2
    layer = layer.crop((left, top, left + w, top + h))
    return Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB")


def squeeze_range(im, headroom):
    """Map 0..255 onto headroom..255-headroom so saturated areas can carry the ID."""
    if headroom <= 0:
        return im
    a = np.asarray(im, dtype=np.float32)
    return Image.fromarray(np.rint(headroom + a * (255 - 2 * headroom) / 255).astype(np.uint8), "RGB")


def realign_crop(published, suspect):
    """Find `suspect` inside `published` as a scaled crop. Return (image, score).

    The image is the suspect scaled and placed where it sits in the published
    frame, on a flat grey canvas, so nothing from our own copy leaks into it.
    Returns (None, score) when the suspect does not look like a crop of it.
    """
    W, H = published.size
    f = 480 / max(W, H)
    small = cv2.cvtColor(cv2.resize(to_bgr(published), (round(W * f), round(H * f)),
                                    interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    sus_gray = cv2.cvtColor(to_bgr(suspect), cv2.COLOR_BGR2GRAY)
    aspect = suspect.height / suspect.width
    best = (-1.0, None, None)                      # score, width fraction, (x, y)
    for frac in np.linspace(0.25, 1.0, 61):        # suspect width as a fraction of W
        tw, th = round(frac * small.shape[1]), round(frac * small.shape[1] * aspect)
        if tw < 24 or th < 24 or th > small.shape[0]:
            continue
        tpl = cv2.resize(sus_gray, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(small, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score > best[0]:
            best = (score, frac, loc)
    score, frac, loc = best
    if frac is None or score < 0.45:
        return None, score

    # Refine at full resolution in a small window around the coarse estimate.
    full = cv2.cvtColor(to_bgr(published), cv2.COLOR_BGR2GRAY)
    margin = int(math.ceil(3 / f))
    fine = (-1.0, None, None, None)
    for fr in np.linspace(frac * 0.985, min(1.0, frac * 1.015), 13):
        tw, th = round(fr * W), round(fr * W * aspect)
        if th > H or tw > W:
            continue
        x0 = max(0, round(loc[0] / f) - margin)
        y0 = max(0, round(loc[1] / f) - margin)
        x1, y1 = min(W, x0 + tw + 2 * margin), min(H, y0 + th + 2 * margin)
        if x1 - x0 < tw or y1 - y0 < th:
            continue
        tpl = cv2.resize(sus_gray, (tw, th), interpolation=cv2.INTER_AREA if tw < suspect.width else cv2.INTER_CUBIC)
        res = cv2.matchTemplate(full[y0:y1, x0:x1], tpl, cv2.TM_CCOEFF_NORMED)
        _, sc, _, l2 = cv2.minMaxLoc(res)
        if sc > fine[0]:
            fine = (sc, (tw, th), x0 + l2[0], y0 + l2[1])
    if fine[1] is None:
        return None, score
    canvas = Image.new("RGB", (W, H), (128, 128, 128))
    canvas.paste(suspect.resize(fine[1], Image.LANCZOS), (fine[2], fine[3]))
    return canvas, fine[0]


def tone_match(published, candidate):
    """Undo a brightness/contrast edit: fit one gain and offset per colour channel.

    Six numbers for the whole image, so this cannot carry the ID across from our
    own copy. Returns None when the two images are not the same picture.
    """
    ref = np.asarray(published, dtype=np.float32)
    cur = np.asarray(candidate, dtype=np.float32)
    out = np.empty_like(cur)
    for c in range(3):
        x, y = cur[..., c].ravel()[::7], ref[..., c].ravel()[::7]
        if x.std() < 1 or y.std() < 1 or np.corrcoef(x, y)[0, 1] < 0.6:
            return None
        gain, offset = np.polyfit(x, y, 1)
        out[..., c] = cur[..., c] * gain + offset
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), "RGB")


def embed_bits(im, bits, key):
    bwm = WaterMark(password_img=key[0], password_wm=key[1])
    bwm.read_img(img=to_bgr(im))
    bwm.read_wm(np.array(bits, dtype=bool), mode="bit")
    return from_bgr(bwm.embed())


def extract_bits(im, key):
    bwm = WaterMark(password_img=key[0], password_wm=key[1])
    raw = bwm.extract(embed_img=to_bgr(im), wm_shape=ID_BITS, mode="bit")
    return [bool(v >= 0.5) for v in np.asarray(raw).ravel()]


def id_to_bits(hex_id):
    n = int(hex_id, 16)
    return [bool((n >> (ID_BITS - 1 - i)) & 1) for i in range(ID_BITS)]


def chance_of(agree, comparisons):
    """Odds that unrelated random bits agree this well in any of the comparisons made."""
    tail = sum(math.comb(ID_BITS, k) for k in range(agree, ID_BITS + 1)) / 2 ** ID_BITS
    return min(1.0, tail * comparisons)


# ----------------------------------------------------------------- embed
def collect_inputs(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT)
        elif p.is_file():
            files.append(p)
        else:
            sys.exit(f"Not found: {p}")
    return files


def cmd_embed(args):
    files = collect_inputs(args.inputs)
    if not files:
        sys.exit("No image files found in the inputs.")
    out = Path(args.out).resolve()
    for f in files:
        if f.resolve().parent == out:
            sys.exit(f"--out is the folder that holds {f}. Refusing: originals must stay untouched.")
    out.mkdir(parents=True, exist_ok=True)
    if args.text is None:
        args.text = f"© {args.owner}"
    key = load_key(create=True)
    entries = load_manifest()
    used = {e["id"] for e in entries}

    for src in files:
        ext = src.suffix.lower()
        out_ext = ext if ext in (".png", ".jpg", ".jpeg", ".webp") else ".png"
        dst = out / (src.stem + out_ext)
        if dst.exists() and not args.force:
            print(f"SKIP  {src.name}: {dst} already exists (use --force to replace it)")
            continue
        im, alpha = open_rgb(src)
        if args.max_size and max(im.size) > args.max_size:
            scale = args.max_size / max(im.size)
            size = (round(im.width * scale), round(im.height * scale))
            im = im.resize(size, Image.LANCZOS)
            alpha = alpha.resize(size, Image.LANCZOS) if alpha else None
        if min(im.size) < MIN_SIDE:
            print(f"SKIP  {src.name}: {im.width}x{im.height} is under {MIN_SIDE}px, too small to hold the ID")
            continue
        if not args.no_visible:
            im = add_visible(im, args.text, args.opacity, args.font)
        while True:
            hex_id = secrets.token_hex(ID_BITS // 8)
            if hex_id not in used:
                break
        marked = embed_bits(squeeze_range(im, args.headroom), id_to_bits(hex_id), key)
        if alpha is not None and out_ext != ".jpg" and out_ext != ".jpeg":
            marked.putalpha(alpha)
        note = f"(c) {datetime.date.today().year} {args.owner}. All rights reserved."
        if out_ext == ".png":
            from PIL.PngImagePlugin import PngInfo
            meta = PngInfo()
            meta.add_text("Copyright", note)
            marked.save(dst, pnginfo=meta, optimize=True)
        else:
            exif = Image.Exif()
            exif[0x8298] = note          # Copyright
            exif[0x013B] = args.owner    # Artist
            marked.save(dst, quality=args.quality, exif=exif)

        # Read the ID back from the file as written before recording it.
        back, _ = open_rgb(dst)
        agree = sum(a == b for a, b in zip(extract_bits(back, key), id_to_bits(hex_id)))
        if agree < MATCH_BITS:
            dst.unlink()
            print(f"FAIL  {src.name}: ID did not survive saving ({agree}/{ID_BITS} bits). No file written.")
            continue
        used.add(hex_id)
        entries = [e for e in entries if e["output"] != str(dst)]
        entries.append({
            "id": hex_id,
            "source": str(src.resolve()),
            "source_sha256": sha256_file(src),
            "output": str(dst),
            "output_sha256": sha256_file(dst),
            "width": marked.width,
            "height": marked.height,
            "visible_mark": None if args.no_visible else args.text,
            "headroom": args.headroom,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        })
        save_manifest(entries)
        print(f"OK    {src.name} -> {dst}  id={hex_id}  readback {agree}/{ID_BITS}")


# ----------------------------------------------------------------- check
def best_match(bits, entries):
    best = (0, None)
    for e in entries:
        agree = sum(a == b for a, b in zip(bits, id_to_bits(e["id"])))
        if agree > best[0]:
            best = (agree, e)
    return best


def cmd_check(args):
    key = load_key(create=False)
    entries = load_manifest()
    if not entries:
        sys.exit(f"{MANIFEST} has no entries, so there is nothing to compare against.")
    sizes = sorted({(e["width"], e["height"]) for e in entries})
    found_any = False

    for path in map(Path, args.images):
        if not path.is_file():
            print(f"{path}: not found")
            continue
        suspect, _ = open_rgb(path)
        best = (0, None, "")
        tries = 0

        # Pass 1: as-is and mirrored, scaled back to each published size.
        for label, img in (("as-is", suspect), ("mirrored", ImageOps.mirror(suspect))):
            for size in sizes:
                ratio = (img.width / img.height) / (size[0] / size[1])
                if not 0.97 < ratio < 1.03:
                    continue
                cand = img if img.size == size else img.resize(size, Image.LANCZOS)
                agree, entry = best_match(extract_bits(cand, key), entries)
                tries += len(entries)
                if agree > best[0]:
                    best = (agree, entry, f"{label}, scaled to {size[0]}x{size[1]}")

        # Pass 1b: same, after undoing a brightness/contrast edit against each published copy.
        if best[0] < MATCH_BITS:
            for label, img in (("as-is", suspect), ("mirrored", ImageOps.mirror(suspect))):
                for e in entries:
                    size = (e["width"], e["height"])
                    ratio = (img.width / img.height) / (size[0] / size[1])
                    if not 0.97 < ratio < 1.03 or not os.path.exists(e["output"]):
                        continue
                    published, _ = open_rgb(e["output"])
                    cand = tone_match(published, img if img.size == size else img.resize(size, Image.LANCZOS))
                    if cand is None:
                        continue
                    bits = extract_bits(cand, key)
                    agree = sum(a == b for a, b in zip(bits, id_to_bits(e["id"])))
                    tries += 1
                    if agree > best[0]:
                        best = (agree, e, f"{label}, brightness/contrast undone against {Path(e['output']).name}")

        # Pass 2: undo cropping by locating the suspect inside each published copy.
        if best[0] < MATCH_BITS and not args.no_crop_search:
            for label, img in (("as-is", suspect), ("mirrored", ImageOps.mirror(suspect))):
                for e in entries:
                    if not os.path.exists(e["output"]):
                        continue
                    published, _ = open_rgb(e["output"])
                    restored, _score = realign_crop(published, img)
                    if restored is None:
                        continue
                    bits = extract_bits(restored, key)
                    if sum(a == b for a, b in zip(bits, id_to_bits(e["id"]))) < MATCH_BITS:
                        # Cropped and tone-edited: fit tone on the overlapping area only.
                        mask = np.any(np.asarray(restored) != 128, axis=2)
                        ys, xs = np.where(mask)
                        if len(ys):
                            box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
                            fixed = tone_match(published.crop(box), restored.crop(box))
                            if fixed is not None:
                                retry = restored.copy()
                                retry.paste(fixed, box[:2])
                                alt = extract_bits(retry, key)
                                if (sum(a == b for a, b in zip(alt, id_to_bits(e["id"])))
                                        > sum(a == b for a, b in zip(bits, id_to_bits(e["id"])))):
                                    bits = alt
                    agree = sum(a == b for a, b in zip(bits, id_to_bits(e["id"])))
                    tries += 1
                    if agree > best[0]:
                        best = (agree, e, f"{label}, re-aligned as a crop of {Path(e['output']).name}")

        agree, entry, how = best
        print(f"\n{path}  ({suspect.width}x{suspect.height})")
        if entry is None or agree < LIKELY_BITS:
            print(f"  NO MATCH. Best agreement {agree}/{ID_BITS} bits, which is what unrelated images give.")
            continue
        found_any = True
        verdict = "MATCH" if agree >= MATCH_BITS else "LIKELY MATCH"
        print(f"  {verdict}: {agree}/{ID_BITS} bits agree with id {entry['id']}  ({how})")
        print(f"  Published copy : {entry['output']}")
        print(f"  Original       : {entry['source']}")
        print(f"  Watermarked on : {entry['created']}")
        print(f"  Odds of this by chance, over {tries} comparisons: about 1 in {1 / chance_of(agree, tries):,.0f}")
    return 0 if found_any else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="write watermarked copies of images")
    e.add_argument("inputs", nargs="+", help="image files or folders of images")
    e.add_argument("--out", required=True, help="folder for the watermarked copies")
    e.add_argument("--owner", required=True, help="name used in the copyright note, e.g. \"Jane Doe Studio\"")
    e.add_argument("--text", help="visible mark text (default: \"© <owner>\")")
    e.add_argument("--font", help="TrueType font file for the visible mark (default: a common system bold font)")
    e.add_argument("--opacity", type=float, default=0.16, help="visible mark opacity, 0 to 1 (default 0.16)")
    e.add_argument("--no-visible", action="store_true", help="hidden ID only, no visible mark")
    e.add_argument("--max-size", type=int, help="shrink so the longest side is at most this many pixels")
    e.add_argument("--quality", type=int, default=92, help="JPEG/WebP quality (default 92)")
    e.add_argument("--headroom", type=int, default=12,
                   help="brightness levels kept free at each end so white/black areas hold the ID (default 12; 0 disables)")
    e.add_argument("--force", action="store_true", help="replace an existing watermarked copy")
    e.set_defaults(func=cmd_embed)

    c = sub.add_parser("check", help="look for a hidden ID in suspect images")
    c.add_argument("images", nargs="+")
    c.add_argument("--no-crop-search", action="store_true", help="skip the slower crop re-alignment pass")
    c.set_defaults(func=cmd_check)

    args = ap.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
