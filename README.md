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
published images it came from. It also bounds how likely an unrelated image is to score that well.

## Install

Python 3.12+ (the pinned NumPy needs it).

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
- Saves each copy to a temporary file first and reads its ID back from that file.
  The destination is replaced only if the ID survived, so a failed save never
  destroys a good copy.
- `--force` replaces an existing copy. The old copy moves to `archive/` in the
  state folder and its manifest row is kept (marked `superseded`), so copies of
  it that are already out in the world still match. Archive names are never
  reused, so one archived copy can't overwrite another.
- The manifest is saved before the new copy replaces the old one. If the
  replace fails, the manifest change is undone, so a file and its record never
  drift apart.
- Published copies get normal file permissions (on Linux and macOS, the mode of
  the copy they replace, or 0666 minus your umask), so a web server can read them.
- Exit code: `0` if no image failed, `1` if any did. Skipped images aren't failures.
- Useful options: `--text` (default `© <owner>`), `--opacity 0.16`, `--no-visible`,
  `--max-size 1600`, `--quality 92`, `--font path/to/font.ttf`, `--force`.
- `--headroom 12` squeezes brightness slightly (white 255 becomes 243) so pure
  white and black areas still carry the ID. Use `0` to turn this off.
- Images under 256 px on the short side are skipped: they're too small to hold the ID.
- Transparency is kept for PNG and WebP (WebP is saved with `exact`, so hidden
  pixels keep their colour). A transparent source that would become a JPEG is
  written as PNG instead. If the ID is weak once the image is flattened onto
  white, `embed` warns you: copies pasted onto a background may not match.
- Several `embed` runs can share one state folder at the same time. They take
  turns writing the manifest and the key.

## Check a suspect image

```bash
.venv/bin/python watermark.py check suspect.jpg
```

A JPEG crop at a different aspect ratio:

```
suspect.jpg  (474x468)
  MATCH: 64/64 bits agree with id 2b0b5513ad946de8  (as-is, re-aligned as a crop of picture.png)
  Published copy : /path/to/published/picture.png
  Original       : /path/to/originals/picture.png
  Watermarked on : 2026-11-20T22:24:32+00:00
  Chance bound   : an unrelated image scores this well in 2 comparisons at most about 1 in 9,223,372,036,854,775,808
                   (random-ID model; not the probability that this attribution is wrong)
```

The verdict needs both enough agreeing bits and a small enough chance bound:

| Verdict | Bits agreeing (of 64) | Chance bound |
|---|---|---|
| MATCH | 58 or more | at most 1 in a million |
| LIKELY MATCH | 52 or more | at most 1 in a thousand |
| NO MATCH | anything else | |

Unrelated images usually score about 32–41 bits.

The chance bound counts every ID comparison `check` made for that image. It
multiplies by that count, so a big manifest needs more bits for the same verdict.
It treats an unrelated image as giving random bits. It bounds a coincidence. It
isn't the probability that the attribution is wrong.

These checks work from the key and manifest alone: resized, recompressed,
mirrored and greyscale images. Cropped or tone-edited images also need your
published copy at the path the manifest records, so the tool can line the
suspect image up against it. A crop must keep at least 25% of the frame's width.
A published copy that is missing, unreadable or a different size is skipped,
with a note. `--no-crop-search` skips that slower pass.

The crop search's correlation gate (0.45) is a heuristic. It hasn't been
calibrated on a large set of real crops.

The exit code is `0` if any image matched and `1` if none did.

## Use it from an AI assistant (MCP)

`mcp_server.py` is a small [MCP](https://modelcontextprotocol.io) server, so an
assistant such as Claude Code can watermark and check images for you. It has three tools:

| Tool | What it does |
|---|---|
| `check_image` | runs `check` on suspect images and returns the report, plus `matched` |
| `watermark_images` | runs `embed` and returns the report, plus `ok` |
| `list_watermarked` | lists the manifest's rows (ID, file paths, date), newest first |

It runs over stdio on your own machine and calls `watermark.py` in a child process,
so it uses the same state folder as the command line. The key is never sent to the
assistant. Only the reports are. Don't expose this server over a network.

Install its extra dependency into the same venv:

```bash
.venv/bin/pip install -r requirements-mcp.txt
```

Then point any MCP client at `.venv/bin/python /full/path/to/mcp_server.py`.

A crop search can take several minutes with a large manifest. The server sends a
progress note every 10 seconds while it works. If your client still times out,
raise its tool timeout (in Claude Code, set `MCP_TOOL_TIMEOUT` in milliseconds).
Cancelling a check stops it. Cancelling a watermark run lets it finish, even if the
assistant then closes, so a file and its manifest row never split.

### Claude Code plugin

The plugin adds the server plus two skills: `/hidden-id-watermark:check-image` and
`/hidden-id-watermark:watermark-images`.

```
/plugin install hidden-id-watermark --marketplace anphonic/hidden-id-watermark
```

The plugin doesn't carry its own copy of the tool, because your key would end up in
the plugin cache. It runs your install, which you name with two environment variables. Set
them in `~/.claude/settings.json`:

```json
{
  "env": {
    "WATERMARK_HOME": "/full/path/to/hidden-id-watermark",
    "WATERMARK_PYTHON": "/full/path/to/hidden-id-watermark/.venv/bin/python"
  }
}
```

On Windows, `WATERMARK_PYTHON` is `...\.venv\Scripts\python.exe`. Add
`WATERMARK_DIR` too if your state folder isn't beside `watermark.py`.

## Your private files

The tool keeps its state beside `watermark.py`. Set `WATERMARK_DIR` to keep it
somewhere else:

| File | What it is | Keep private? | Back up? |
|---|---|---|---|
| `watermark.key` | secret embedding key, created on the first `embed` | **yes** | **yes**: without it no ID can ever be read |
| `manifest.json` | which ID went into which image, with file paths and hashes | yes: it records your folder paths | yes, after every batch |
| `archive/` | published copies that `--force` replaced | yes | yes: `check` aligns old copies against them |

The key is created readable only by you: mode 600 on Linux and macOS, and on
Windows an ACL with inheritance removed that grants only your account. Every
run warns you if other accounts can read it.

All of these are in `.gitignore`. Don't commit them. With the key, someone could read
your IDs and also forge marks of their own.

Keep your unwatermarked originals somewhere separate and untouched. A match
shows that a suspect image came from one of your published copies. The originals
show that you made it first.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The tests run in a throwaway `WATERMARK_DIR`, so they never touch your real key.
The MCP server's tests are skipped unless `requirements-mcp.txt` is installed.

## Limits

- An invisible watermark is evidence, not DRM. Heavy edits (strong blur,
  rotation, collage, AI regeneration) can destroy it.
- Only you can run `check` (it needs the key), so it proves things to you, and
  through you to others. It isn't a public verification system.

## License

MIT. See [LICENSE](LICENSE).
