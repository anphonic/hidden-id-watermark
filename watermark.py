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
       Each copy is written to a temporary file and its ID is read back first.
       It replaces anything at the destination only if that check passes. With
       --force, the copy it replaces is moved to archive/ and its manifest row
       is kept, so copies already out in the world still match.
       Exit code: 0 if no image failed, 1 if any did.

check  Reads the hidden ID out of a suspect image and compares it with every ID
       in manifest.json. It tries the image as-is and mirrored, then with a
       brightness/contrast edit undone, then re-aligned against each published
       copy to undo cropping. The last two need the published copies to still be
       at the paths recorded in manifest.json. It reports how many of the 64 bits
       agree and a bound on how likely an unrelated image is to score that well.
       Exit code: 0 if any image matched, 1 if none did.

Files that must be kept private and backed up. They live beside this script,
or in the folder named by the WATERMARK_DIR environment variable:
    watermark.key   secret embedding key. Without it no ID can be read back.
    manifest.json   which ID went into which image.
    archive/        published copies that --force replaced.
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from PIL.PngImagePlugin import PngInfo

import cv2
import blind_watermark
from blind_watermark import WaterMark

if hasattr(blind_watermark, "bw_notes"):
    blind_watermark.bw_notes.close()

HERE = Path(os.environ.get("WATERMARK_DIR") or Path(__file__).resolve().parent).resolve()

ID_BITS = 64
MATCH_BITS = 58        # MATCH needs at least this many of 64 bits agreeing ...
MATCH_CHANCE = 1e-6    # ... and a chance bound no larger than this
LIKELY_BITS = 52       # LIKELY MATCH needs at least this many ...
LIKELY_CHANCE = 1e-3   # ... and a chance bound no larger than this
MIN_SIDE = 256         # smaller images do not hold the ID reliably
ALIGN_MIN_SCORE = 0.45 # crop re-alignment correlation gate (heuristic, not calibrated)
FLAT_STD = 2.0         # a suspect this flat has nothing to align on
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
BROAD_PRINCIPALS = ("everyone", "builtin\\users", "authenticated users")
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


# ----------------------------------------------------------------- state files
def key_file():
    return HERE / "watermark.key"


def manifest_file():
    return HERE / "manifest.json"


def archive_dir():
    return HERE / "archive"


def lock_file():
    return HERE / ".watermark.lock"


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


@contextlib.contextmanager
def state_lock():
    """Exclusive lock on the state folder, so concurrent runs don't lose rows."""
    HERE.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_file(), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)   # itself retries for ~10 s
                    break
                except OSError:
                    time.sleep(0.2)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        os.close(fd)   # also releases flock


def replace_file(src, dst):
    """os.replace, retried briefly: on Windows a reader holding dst blocks it."""
    for attempt in range(50):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 49:
                raise
            time.sleep(0.1)


def write_private_file(path, text, restrict=False):
    """Write text to path atomically: unique temp file, fsync, then replace."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if restrict:
            restrict_to_owner(tmp)
        replace_file(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ----------------------------------------------------------------- key
def restrict_to_owner(path):
    """Make path readable by the current user only."""
    if os.name != "nt":
        os.chmod(path, 0o600)
        return
    try:
        user = subprocess.run(["whoami"], capture_output=True, text=True, check=True).stdout.strip()
        subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"],
                       capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"WARNING: could not restrict {path} to your account ({exc}). "
              "Remove other users' access to it by hand.", file=sys.stderr)


def warn_if_key_exposed(path):
    """Warn when other accounts can read the key."""
    try:
        if os.name != "nt":
            if path.stat().st_mode & 0o077:
                print(f"WARNING: {path} is readable by other users. Run: chmod 600 {path}", file=sys.stderr)
            return
        acl = subprocess.run(["icacls", str(path)], capture_output=True, text=True).stdout.lower()
        if any(p in acl for p in BROAD_PRINCIPALS):
            print(f"WARNING: {path} is readable by other accounts on this PC. Restrict it with:\n"
                  f'  icacls "{path}" /inheritance:r /grant:r "%USERNAME%:F"', file=sys.stderr)
    except OSError:
        pass


def load_key(create):
    """Return (password_img, password_wm). The file is parsed, never executed."""
    kf = key_file()
    if not kf.exists():
        if not create:
            sys.exit(f"No key file at {kf}. Nothing has been watermarked with this tool yet.")
        with state_lock():
            if not kf.exists():          # another run may have made it while we waited
                write_private_file(kf, f"password_img={secrets.randbelow(2**31 - 2) + 1}\n"
                                       f"password_wm={secrets.randbelow(2**31 - 2) + 1}\n", restrict=True)
                print(f"Created new secret key: {kf}  (back this file up)")
    warn_if_key_exposed(kf)
    values = {}
    for line in kf.read_text().splitlines():
        line = line.rstrip("\r")
        for name in ("password_img", "password_wm"):
            if line.startswith(name + "="):
                values[name] = int(line[len(name) + 1:])
    if set(values) != {"password_img", "password_wm"}:
        sys.exit(f"{kf} is missing password_img or password_wm.")
    return values["password_img"], values["password_wm"]


# ----------------------------------------------------------------- manifest
def load_manifest():
    mf = manifest_file()
    if not mf.exists():
        return []
    return json.loads(mf.read_text(encoding="utf-8"))


def save_manifest(entries):
    """Call with state_lock() held."""
    write_private_file(manifest_file(), json.dumps(entries, indent=2) + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------- image helpers
def open_image(path):
    """Return (RGB image, alpha channel or None), EXIF rotation applied.

    Alpha is None when the image has no transparency at all (including an
    alpha channel that is fully opaque). The file is closed before returning,
    so it can be replaced right after, even on Windows.
    """
    with Image.open(path) as raw:
        im = ImageOps.exif_transpose(raw)     # returns a loaded copy
    if im.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in im.info:
        rgba = im.convert("RGBA")
        alpha = rgba.getchannel("A")
        if alpha.getextrema() == (255, 255):
            alpha = None
        return rgba.convert("RGB"), alpha
    return im.convert("RGB"), None


def to_bgr(im):
    return np.ascontiguousarray(np.asarray(im, dtype=np.uint8)[:, :, ::-1])


def from_bgr(arr):
    arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(arr[:, :, ::-1]), "RGB")


def to_gray(im):
    return cv2.cvtColor(to_bgr(im), cv2.COLOR_BGR2GRAY)


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
    """Tile semi-transparent diagonal text over the whole image.

    One stamp is rendered and rotated once, then composited at points of a
    rotated lattice. Memory stays proportional to the image, not its diagonal
    squared, so long panoramas are fine.
    """
    w, h = im.size
    size = max(14, round(min(w, h) / 22))
    font = find_font(size, font_path)
    scalable = isinstance(font, ImageFont.FreeTypeFont)
    stroke = max(1, round(getattr(font, "size", size) / 22)) if scalable else 0
    a = round(255 * opacity)

    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    plain = probe.textbbox((0, 0), text, font=font)
    box = probe.textbbox((0, 0), text, font=font, stroke_width=stroke)
    tw, th = plain[2] - plain[0], plain[3] - plain[1]
    stamp = Image.new("RGBA", (box[2] - box[0] + 2, box[3] - box[1] + 2), (0, 0, 0, 0))
    ImageDraw.Draw(stamp).text((1 - box[0], 1 - box[1]), text, font=font, fill=(255, 255, 255, a),
                               stroke_width=stroke, stroke_fill=(0, 0, 0, a))
    angle = 30
    rot = stamp.rotate(angle, resample=Image.BICUBIC, expand=True)
    rw, rh = rot.size

    # Lattice laid out on a square of side S (the image diagonal), rotated about
    # its centre, then viewed through the image-sized window in the middle.
    side = int(math.hypot(w, h)) + 4
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    half = side / 2
    off_x, off_y = (side - w) // 2, (side - h) // 2
    step_x, step_y = max(1, int(tw * 1.45)), max(1, int(th * 5.0))
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for row, y in enumerate(range(0, side, step_y)):
        for x in range(-step_x if row % 2 else -step_x // 2, side, step_x):
            dx, dy = x + stamp.width / 2 - half, y + stamp.height / 2 - half
            cx = half + dx * c + dy * s - off_x          # counter-clockwise, y down
            cy = half - dx * s + dy * c - off_y
            px, py = round(cx - rw / 2), round(cy - rh / 2)
            x0, y0 = max(0, px), max(0, py)
            x1, y1 = min(w, px + rw), min(h, py + rh)
            if x1 <= x0 or y1 <= y0:
                continue
            layer.alpha_composite(rot, dest=(x0, y0), source=(x0 - px, y0 - py, x1 - px, y1 - py))
    return Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB")


def squeeze_range(im, headroom):
    """Map 0..255 onto headroom..255-headroom so saturated areas can carry the ID."""
    if headroom <= 0:
        return im
    a = np.asarray(im, dtype=np.float32)
    return Image.fromarray(np.rint(headroom + a * (255 - 2 * headroom) / 255).astype(np.uint8), "RGB")


def resize_rgba(im, alpha, size):
    """Resize colour and alpha together, premultiplied so edges don't pick up a fringe."""
    if alpha is None:
        return im.resize(size, Image.LANCZOS), None
    rgba = im.copy()
    rgba.putalpha(alpha)
    rgba = rgba.convert("RGBa").resize(size, Image.LANCZOS).convert("RGBA")
    return rgba.convert("RGB"), rgba.getchannel("A")


# ----------------------------------------------------------------- crop alignment
def realign_crop(published, suspect):
    """Find `suspect` inside `published` as a scaled crop.

    Returns (image, score, box). The image is the suspect scaled and placed where
    it sits in the published frame, on a flat grey canvas, so nothing from our
    own copy leaks into it; box is (left, top, right, bottom) of that placement.
    Returns (None, score, None) when it does not look like a crop.

    The search covers crops from 25% up to the largest the suspect's aspect ratio
    allows. The position is chosen by image correlation only, never by ID bits.
    """
    W, H = published.size
    sus_gray = to_gray(suspect)
    if sus_gray.std() < FLAT_STD:
        return None, 0.0, None
    aspect = suspect.height / suspect.width
    s_max = min(1.0, H / (W * aspect))           # widest crop (as a fraction of W) that fits
    s_min = 0.25 * s_max
    f = min(1.0, max(480 / max(W, H), 24 / (s_min * W * min(1.0, aspect))))
    small = cv2.resize(to_gray(published), (max(1, round(W * f)), max(1, round(H * f))),
                       interpolation=cv2.INTER_AREA)
    sh, sw = small.shape

    peaks = []                                   # (score, width fraction, (x, y) in small)
    for frac in np.linspace(s_min, s_max, 61):
        tw, th = round(frac * sw), round(frac * sw * aspect)
        if tw < 8 or th < 8 or tw > sw or th > sh:
            continue
        tpl = cv2.resize(sus_gray, (tw, th), interpolation=cv2.INTER_AREA)
        if tpl.std() < FLAT_STD:
            continue
        _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(small, tpl, cv2.TM_CCOEFF_NORMED))
        peaks.append((score, float(frac), loc))
    if not peaks:
        return None, 0.0, None
    peaks.sort(key=lambda p: -p[0])
    chosen = []
    for p in peaks:                              # up to 3 distinct candidates
        if all(abs(p[1] - q[1]) / q[1] > 0.04 or math.hypot(p[2][0] - q[2][0], p[2][1] - q[2][1]) > 8
               for q in chosen):
            chosen.append(p)
        if len(chosen) == 3:
            break

    full = to_gray(published)
    margin = int(math.ceil(4 / f)) + 2

    def match_at(tw, loc):
        th = round(tw * aspect)
        if tw < 8 or th < 8 or tw > W or th > H:
            return None
        cx, cy = int(loc[0] / f), int(loc[1] / f)
        x0 = min(max(0, cx - margin), W - tw)
        y0 = min(max(0, cy - margin), H - th)
        x1 = min(W, max(x0 + tw, cx + tw + margin + 1))
        y1 = min(H, max(y0 + th, cy + th + margin + 1))
        interp = cv2.INTER_AREA if tw < suspect.width else cv2.INTER_CUBIC
        tpl = cv2.resize(sus_gray, (tw, th), interpolation=interp)
        _, sc, _, l2 = cv2.minMaxLoc(cv2.matchTemplate(full[y0:y1, x0:x1], tpl, cv2.TM_CCOEFF_NORMED))
        return sc, (tw, th), (x0 + l2[0], y0 + l2[1])

    best = None
    for _score, frac, loc in chosen:
        local = None
        for fr in np.linspace(frac * 0.985, frac * 1.015, 13):
            r = match_at(round(fr * W), loc)
            if r and (local is None or r[0] > local[0]):
                local = r
        if local is None:
            continue
        step = max(1, round(local[1][0] * 0.004))
        while True:                              # integer width refinement, halving stride
            improved = False
            for tw in (local[1][0] - step, local[1][0] + step):
                r = match_at(tw, loc)
                if r and r[0] > local[0]:
                    local, improved = r, True
            if not improved:
                if step == 1:
                    break
                step = max(1, step // 2)
        if best is None or local[0] > best[0]:
            best = local
    if best is None or best[0] < ALIGN_MIN_SCORE:
        return None, (best[0] if best else peaks[0][0]), None
    score, (tw, th), (x, y) = best
    canvas = Image.new("RGB", (W, H), (128, 128, 128))
    canvas.paste(suspect.resize((tw, th), Image.LANCZOS), (x, y))
    return canvas, score, (x, y, x + tw, y + th)


def tone_match(published, candidate):
    """Undo a brightness/contrast edit: fit one gain and offset per colour channel.

    Six numbers for the whole image, so this cannot carry the ID across from our
    own copy. Returns None when the two images are not the same picture.
    """
    ref = np.asarray(published, dtype=np.float32)
    cur = np.asarray(candidate, dtype=np.float32)
    if ref.shape != cur.shape:
        return None
    out = np.empty_like(cur)
    for c in range(3):
        x, y = cur[..., c].ravel()[::7], ref[..., c].ravel()[::7]
        if x.size < 16 or x.std() < 1 or y.std() < 1 or np.corrcoef(x, y)[0, 1] < 0.6:
            return None
        gain, offset = np.polyfit(x, y, 1)
        out[..., c] = cur[..., c] * gain + offset
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), "RGB")


# ----------------------------------------------------------------- ID bits and odds
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


def agreement(bits, hex_id):
    return sum(a == b for a, b in zip(bits, id_to_bits(hex_id)))


def chance_of(agree, comparisons):
    """Union bound: chance that random bits agree this well in any of `comparisons` tries.

    This models an unrelated image as giving random bits. It bounds a chance
    coincidence; it is not the probability that an attribution is wrong.
    """
    tail = sum(math.comb(ID_BITS, k) for k in range(agree, ID_BITS + 1)) / 2 ** ID_BITS
    return min(1.0, tail * max(1, comparisons))


def verdict(agree, chance):
    if agree >= MATCH_BITS and chance <= MATCH_CHANCE:
        return "MATCH"
    if agree >= LIKELY_BITS and chance <= LIKELY_CHANCE:
        return "LIKELY MATCH"
    return "NO MATCH"


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


def save_image(img, path, fmt, owner, quality):
    note = f"(c) {datetime.date.today().year} {owner}. All rights reserved."
    if fmt == "PNG":
        meta = PngInfo()
        meta.add_text("Copyright", note)
        img.save(path, format="PNG", pnginfo=meta, optimize=True)
        return
    exif = Image.Exif()
    exif[0x8298] = note          # Copyright
    exif[0x013B] = owner         # Artist
    if fmt == "WEBP":
        img.save(path, format="WEBP", quality=quality, exif=exif, exact=img.mode == "RGBA")
    else:
        img.save(path, format="JPEG", quality=quality, exif=exif)


def verify_written(path, hex_id, key, alpha):
    """Read the file back. Return (ok, readback agreement, list of warnings/errors)."""
    back, back_alpha = open_image(path)
    agree = agreement(extract_bits(back, key), hex_id)
    msgs = []
    if agree < MATCH_BITS:
        return False, agree, [f"ID did not survive saving ({agree}/{ID_BITS} bits)"]
    if alpha is not None:
        if back_alpha is None:
            return False, agree, ["transparency was lost when saving"]
        diff = np.abs(np.asarray(back_alpha, np.int16) - np.asarray(alpha, np.int16)).max()
        if diff > 1:
            msgs.append(f"WARN transparency changed by up to {diff} levels when saving")
        flat = Image.new("RGB", back.size, (255, 255, 255))
        flat.paste(back, mask=back_alpha)
        flat_agree = agreement(extract_bits(flat, key), hex_id)
        if flat_agree < MATCH_BITS:
            msgs.append(f"WARN ID is weak once flattened onto white ({flat_agree}/{ID_BITS} bits): "
                        "copies pasted onto a background may not match")
    return True, agree, msgs


def archive_replaced(dst, entries, new_id):
    """Keep the copy at dst that is about to be replaced. Call with state_lock() held."""
    live = [e for e in entries if e.get("output") == str(dst) and not e.get("superseded")]
    archive_dir().mkdir(parents=True, exist_ok=True)
    if live:
        target = archive_dir() / (live[-1]["id"] + dst.suffix.lower())
    else:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = archive_dir() / f"untracked-{stamp}-{dst.name}"
    shutil.copy2(dst, target)
    when = now_utc()
    for e in live:
        e["output"] = str(target)
        e["superseded"] = when
        e["superseded_by"] = new_id
    return target


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
    used = {e["id"] for e in load_manifest()}
    failures = 0

    for src in files:
        ext = src.suffix.lower()
        out_ext = ext if ext in (".png", ".jpg", ".jpeg", ".webp") else ".png"
        try:
            im, alpha = open_image(src)
        except Exception as exc:
            print(f"FAIL  {src.name}: cannot read it as an image ({exc})")
            failures += 1
            continue
        if alpha is not None and out_ext in (".jpg", ".jpeg"):
            out_ext = ".png"
            print(f"NOTE  {src.name}: has transparency, which JPEG cannot store; writing PNG")
        dst = out / (src.stem + out_ext)
        if dst.exists() and not args.force:
            print(f"SKIP  {src.name}: {dst} already exists (use --force to replace it)")
            continue
        if args.max_size and max(im.size) > args.max_size:
            scale = args.max_size / max(im.size)
            im, alpha = resize_rgba(im, alpha, (round(im.width * scale), round(im.height * scale)))
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
        if alpha is not None:
            marked.putalpha(alpha)
        fmt = {".png": "PNG", ".webp": "WEBP"}.get(out_ext, "JPEG")

        fd, tmp = tempfile.mkstemp(dir=out, prefix=f".{src.stem}.", suffix=out_ext)
        os.close(fd)
        try:
            save_image(marked, tmp, fmt, args.owner, args.quality)
            ok, agree, msgs = verify_written(tmp, hex_id, key, alpha)
            if not ok:
                failures += 1
                kept = f"{dst} left as it was" if dst.exists() else "no file written"
                print(f"FAIL  {src.name}: {msgs[0]}. {kept}.")
                continue
            with state_lock():
                entries = load_manifest()      # re-read: another run may have added rows
                if hex_id in {e["id"] for e in entries}:
                    failures += 1
                    print(f"FAIL  {src.name}: id {hex_id} was taken by another run meanwhile; try again")
                    continue
                archived = None
                if dst.exists():
                    if not args.force:
                        print(f"SKIP  {src.name}: {dst} appeared while working (use --force to replace it)")
                        continue
                    archived = archive_replaced(dst, entries, hex_id)
                output_sha = sha256_file(tmp)
                replace_file(tmp, dst)
                entries.append({
                    "id": hex_id,
                    "source": str(src.resolve()),
                    "source_sha256": sha256_file(src),
                    "output": str(dst),
                    "output_sha256": output_sha,
                    "width": marked.width,
                    "height": marked.height,
                    "visible_mark": None if args.no_visible else args.text,
                    "headroom": args.headroom,
                    "created": now_utc(),
                })
                save_manifest(entries)
            used.add(hex_id)
            print(f"OK    {src.name} -> {dst}  id={hex_id}  readback {agree}/{ID_BITS}")
            if archived:
                print(f"      previous copy kept at {archived}")
            for m in msgs:
                print(f"      {m}")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return 1 if failures else 0


# ----------------------------------------------------------------- check
def best_match(bits, entries):
    best = (0, None)
    for e in entries:
        agree = agreement(bits, e["id"])
        if agree > best[0]:
            best = (agree, e)
    return best


def load_reference(entry, cache):
    """Return (published image or None, note or None) for an entry, cached by path."""
    path = entry.get("output")
    if path in cache:
        return cache[path]
    ref, note = None, None
    if not path or not os.path.exists(path):
        note = f"published copy missing, skipped for alignment: {path}"
    else:
        try:
            img, _ = open_image(path)
            if img.size != (entry["width"], entry["height"]):
                note = (f"{path} is {img.width}x{img.height} but was published at "
                        f"{entry['width']}x{entry['height']}; skipped for alignment")
            else:
                ref = img
                if entry.get("output_sha256") and sha256_file(path) != entry["output_sha256"]:
                    note = f"{path} has changed since it was published; used for alignment only"
        except Exception as exc:
            note = f"cannot read published copy {path} ({exc}); skipped for alignment"
    cache[path] = (ref, note)
    return cache[path]


def fit(img, size):
    return img if img.size == size else img.resize(size, Image.LANCZOS)


def same_aspect(a, b):
    return 0.97 < (a[0] / a[1]) / (b[0] / b[1]) < 1.03


def cmd_check(args):
    key = load_key(create=False)
    entries = load_manifest()
    if not entries:
        sys.exit(f"{manifest_file()} has no entries, so there is nothing to compare against.")
    sizes = sorted({(e["width"], e["height"]) for e in entries})
    refs = {}
    found_any = False

    for path in map(Path, args.images):
        if not path.is_file():
            print(f"{path}: not found")
            continue
        try:
            suspect, _ = open_image(path)
        except Exception as exc:
            print(f"{path}: cannot read it as an image ({exc})")
            continue
        notes = []

        def reference(e):
            ref, note = load_reference(e, refs)
            if note and note not in notes:
                notes.append(note)
            return ref

        best = (0, None, "")
        tries = 0               # every ID comparison made, for the chance bound
        flips = (("as-is", suspect), ("mirrored", ImageOps.mirror(suspect)))

        # Pass 1: as-is and mirrored, scaled back to each published size.
        for label, img in flips:
            for size in sizes:
                if not same_aspect(img.size, size):
                    continue
                agree, entry = best_match(extract_bits(fit(img, size), key), entries)
                tries += len(entries)
                if agree > best[0]:
                    best = (agree, entry, f"{label}, scaled to {size[0]}x{size[1]}")

        # Pass 1b: same, after undoing a brightness/contrast edit against each published copy.
        if best[0] < MATCH_BITS:
            for label, img in flips:
                for e in entries:
                    size = (e["width"], e["height"])
                    if not same_aspect(img.size, size):
                        continue
                    published = reference(e)
                    if published is None:
                        continue
                    cand = tone_match(published, fit(img, size))
                    if cand is None:
                        continue
                    agree = agreement(extract_bits(cand, key), e["id"])
                    tries += 1
                    if agree > best[0]:
                        best = (agree, e, f"{label}, brightness/contrast undone against {Path(e['output']).name}")

        # Pass 2: undo cropping by locating the suspect inside each published copy.
        if best[0] < MATCH_BITS and not args.no_crop_search:
            for label, img in flips:
                for e in entries:
                    published = reference(e)
                    if published is None:
                        continue
                    restored, _score, box = realign_crop(published, img)
                    if restored is None:
                        continue
                    agree = agreement(extract_bits(restored, key), e["id"])
                    tries += 1
                    how = f"{label}, re-aligned as a crop of {Path(e['output']).name}"
                    if agree < MATCH_BITS:
                        # Cropped and tone-edited: fit tone on the overlapping area only.
                        fixed = tone_match(published.crop(box), restored.crop(box))
                        if fixed is not None:
                            retry = restored.copy()
                            retry.paste(fixed, box[:2])
                            alt = agreement(extract_bits(retry, key), e["id"])
                            tries += 1
                            if alt > agree:
                                agree, how = alt, how + ", brightness/contrast undone"
                    if agree > best[0]:
                        best = (agree, e, how)

        agree, entry, how = best
        chance = chance_of(agree, tries)
        result = verdict(agree, chance) if entry else "NO MATCH"
        print(f"\n{path}  ({suspect.width}x{suspect.height})")
        for n in notes:
            print(f"  note: {n}")
        if result == "NO MATCH":
            if agree >= LIKELY_BITS:
                print(f"  NO MATCH. Best agreement {agree}/{ID_BITS} bits, but over {tries} comparisons "
                      f"that could happen by chance (about 1 in {1 / chance:,.0f}).")
            else:
                print(f"  NO MATCH. Best agreement {agree}/{ID_BITS} bits, which is what unrelated images give.")
            continue
        found_any = True
        print(f"  {result}: {agree}/{ID_BITS} bits agree with id {entry['id']}  ({how})")
        print(f"  Published copy : {entry['output']}")
        if entry.get("superseded"):
            print(f"  Superseded     : {entry['superseded']} by id {entry.get('superseded_by')}")
        print(f"  Original       : {entry['source']}")
        print(f"  Watermarked on : {entry['created']}")
        print(f"  Chance bound   : an unrelated image scores this well in {tries} comparisons "
              f"at most about 1 in {1 / chance:,.0f}")
        print("                   (random-ID model; not the probability that this attribution is wrong)")
    return 0 if found_any else 1


def main(argv=None):
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
    e.add_argument("--force", action="store_true",
                   help="replace an existing watermarked copy (the old one is kept in archive/)")
    e.set_defaults(func=cmd_embed)

    c = sub.add_parser("check", help="look for a hidden ID in suspect images")
    c.add_argument("images", nargs="+")
    c.add_argument("--no-crop-search", action="store_true", help="skip the slower crop re-alignment pass")
    c.set_defaults(func=cmd_check)

    args = ap.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
