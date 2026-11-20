# hidden-id-watermark

Mark the images you publish so you can prove later that a copy came from you.

Each watermarked copy gets three layers:

1. **A visible mark**: tiled, semi-transparent diagonal text (optional).
2. **A hidden 64-bit ID** spread through the pixels with a keyed DWT-DCT-SVD
   watermark ([blind-watermark](https://github.com/guofei9987/blind_watermark)).
   You can't see it, and it survives resizing, JPEG recompression, mirroring,
   greyscale, brightness/contrast edits and cropping.
3. **A copyright note** in the file metadata. Anyone can strip it, so it's only a courtesy.

`check` reads the ID back from a suspect image and tells you which of your
published images it came from. It also gives the odds of that match happening by chance.

## Install

Python 3.10+ (built and tested on 3.12).

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
```

Keep the pinned versions. A different `blind-watermark` version may read IDs
differently, and then IDs embedded with this version may stop matching.

## Watermark images

```bash
.venv/bin/python watermark.py embed photos/ --out published/ --owner "Jane Doe Studio"
```

- Writes **copies** into `--out` and refuses to write into a folder that holds an input.
- Reads every ID back from the saved file and records it only if it survived saving.
- Useful options: `--text` (default `© <owner>`), `--opacity 0.16`, `--no-visible`,
  `--max-size 1600`, `--quality 92`, `--font path/to/font.ttf`, `--force`.
- `--headroom 12` squeezes brightness slightly (white 255 becomes 243) so pure
  white and black areas still carry the ID. Use `0` to turn this off.
- Images under 256 px on the short side are skipped: they're too small to hold the ID.

## Check a suspect image

```bash
.venv/bin/python watermark.py check suspect.jpg
```

```
suspect.jpg  (640x480)
  MATCH: 63/64 bits agree with id 3f9c...  (as-is, scaled to 800x600)
  Published copy : /path/to/published/picture.png
  ...
  Odds of this by chance, over 12 comparisons: about 1 in 1,537,228,672,809,129
```

| Bits agreeing (of 64) | Meaning |
|---|---|
| 58 or more | match |
| 52–57 | likely match, reported with its odds |
| about 32–41 | what unrelated images score |

These checks work from the key and manifest alone: resized, recompressed,
mirrored and greyscale images. Cropped or tone-edited images also need your
published copy at the path the manifest records, so the tool can line the
suspect image up against it. `--no-crop-search` skips that slower pass.

The exit code is `0` if any image matched and `1` if none did.

## Your private files

The tool keeps two files beside `watermark.py`. Set `WATERMARK_DIR` to keep them
somewhere else:

| File | What it is | Keep private? | Back up? |
|---|---|---|---|
| `watermark.key` | secret embedding key, created on the first `embed` | **yes** | **yes**: without it no ID can ever be read |
| `manifest.json` | which ID went into which image, with file paths and hashes | yes: it records your folder paths | yes, after every batch |

Both are in `.gitignore`. Don't commit them. With the key, someone could read
your IDs and also forge marks of their own.

Keep your unwatermarked originals somewhere separate and untouched. A match
shows that a suspect image came from one of your published copies. The originals
show that you made it first.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The test runs in a throwaway `WATERMARK_DIR`, so it never touches your real key.

## Limits

- An invisible watermark is evidence, not DRM. Heavy edits (strong blur,
  rotation, collage, AI regeneration) can destroy it.
- Only you can run `check` (it needs the key), so it proves things to you, and
  through you to others. It isn't a public verification system.

## License

MIT. See [LICENSE](LICENSE).
