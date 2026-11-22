#!/usr/bin/env python3
"""MCP server for hidden-id-watermark.

Lets an AI assistant run `watermark.py` for you:
    check_image          read the hidden ID back from suspect images
    watermark_images     write watermarked copies (the `embed` command)
    list_watermarked     list what the manifest records

It runs over stdio only, on your own machine, as you. Your key never leaves
this computer and is never sent to the assistant: the tools return the same
report the command line prints. Each call runs watermark.py in a child process
with the Python that runs this server, so the state folder is the same one the
command line uses (beside watermark.py, or WATERMARK_DIR).

    python mcp_server.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Annotated

# This tool works offline: no update check against PyPI at startup.
os.environ.setdefault("FASTMCP_CHECK_FOR_UPDATES", "off")

from fastmcp import Context, FastMCP  # noqa: E402
from pydantic import Field  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "watermark.py"
HEARTBEAT_SECONDS = 10

mcp = FastMCP(
    name="hidden-id-watermark",
    instructions=(
        "Tools for the user's own image watermarks. check_image tells whether a suspect "
        "image came from one of the user's published copies; a crop search can take "
        "several minutes. watermark_images writes new watermarked copies. Report verdicts "
        "exactly as returned: LIKELY MATCH is not MATCH, and the chance bound is not the "
        "probability that the attribution is wrong. Never try to read watermark.key."
    ),
)


def absolute(paths):
    """Resolve paths so a name like '-x' can't be read as an option."""
    return [str(Path(p).expanduser().resolve()) for p in paths]


async def run_watermark(args, ctx, kill_on_cancel):
    """Run watermark.py, sending a progress note every few seconds while it works."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(SCRIPT), *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    started = time.monotonic()
    done = asyncio.ensure_future(proc.communicate())
    try:
        while True:
            finished, _ = await asyncio.wait({done}, timeout=HEARTBEAT_SECONDS)
            if finished:
                break
            elapsed = int(time.monotonic() - started)
            await ctx.report_progress(elapsed, None, f"still working ({elapsed} s)")
    except BaseException:
        # Cancelled or failed. A check only reads, so stop it rather than leave
        # a crop search running. An embed is left to finish: killing it between
        # the manifest save and the file replace would split a file from its row.
        if kill_on_cancel and proc.returncode is None:
            proc.kill()
            done.cancel()
        raise
    out, err = done.result()
    return {
        "exit_code": proc.returncode,
        "report": out.decode("utf-8", "replace").strip(),
        "errors": err.decode("utf-8", "replace").strip(),
        "seconds": round(time.monotonic() - started, 1),
    }


@mcp.tool(annotations={"title": "Check suspect images", "read_only_hint": True, "open_world_hint": False})
async def check_image(
    paths: Annotated[list[str], Field(min_length=1, description="Suspect image files on this computer.")],
    ctx: Context,
    crop_search: Annotated[bool, Field(
        description="Also try re-aligning the image as a crop of each published copy. "
                    "Slower (minutes with a big manifest) but needed for cropped or tone-edited copies.")] = True,
) -> dict:
    """Read the hidden ID from suspect images and say which published copy each came from.

    `matched` is true if any image got a MATCH or LIKELY MATCH verdict. `report` holds
    the full per-image verdicts, the matching file paths and the chance bound.
    """
    args = ["check"] + ([] if crop_search else ["--no-crop-search"]) + ["--", *absolute(paths)]
    result = await run_watermark(args, ctx, kill_on_cancel=True)
    result["matched"] = result["exit_code"] == 0
    return result


@mcp.tool(annotations={"title": "Watermark images", "read_only_hint": False,
                       "destructive_hint": False, "open_world_hint": False})
async def watermark_images(
    inputs: Annotated[list[str], Field(min_length=1, description="Image files or folders of images.")],
    out: Annotated[str, Field(description="Folder for the watermarked copies. Must not hold any input.")],
    owner: Annotated[str, Field(min_length=1, description='Name for the copyright note, e.g. "Jane Doe Studio".')],
    ctx: Context,
    text: Annotated[str | None, Field(description='Visible mark text. Default "© <owner>".')] = None,
    visible: Annotated[bool, Field(description="Add the tiled visible mark.")] = True,
    opacity: Annotated[float, Field(ge=0, le=1, description="Visible mark opacity.")] = 0.16,
    max_size: Annotated[int | None, Field(ge=256, description="Shrink so the longest side is at most this.")] = None,
    quality: Annotated[int, Field(ge=1, le=100, description="JPEG/WebP quality.")] = 92,
    headroom: Annotated[int, Field(ge=0, le=64, description="Brightness levels kept free at each end.")] = 12,
    force: Annotated[bool, Field(
        description="Replace copies that already exist in `out`. The old copies move to archive/ "
                    "and keep matching. Only set this when the user asked to replace them.")] = False,
) -> dict:
    """Write watermarked copies of images (visible mark, hidden 64-bit ID, copyright note).

    Originals are never changed. Without `force`, existing copies are skipped. `ok` is
    true if no image failed; `report` lists what happened to each one.
    """
    args = ["embed", "--out", absolute([out])[0], "--owner", owner,
            "--opacity", str(opacity), "--quality", str(quality), "--headroom", str(headroom)]
    if text is not None:
        args += ["--text", text]
    if not visible:
        args.append("--no-visible")
    if max_size is not None:
        args += ["--max-size", str(max_size)]
    if force:
        args.append("--force")
    result = await run_watermark(args + ["--", *absolute(inputs)], ctx, kill_on_cancel=False)
    result["ok"] = result["exit_code"] == 0
    return result


@mcp.tool(annotations={"title": "List watermarked images", "read_only_hint": True, "open_world_hint": False})
def list_watermarked(
    contains: Annotated[str | None, Field(description="Only rows whose file path contains this text.")] = None,
    include_superseded: Annotated[bool, Field(description="Include copies that --force replaced.")] = False,
    limit: Annotated[int, Field(ge=1, le=1000)] = 100,
) -> dict:
    """List the watermarked copies the manifest records, newest first."""
    sys.path.insert(0, str(SCRIPT.parent))
    import watermark  # local module; imported lazily so the server starts fast

    rows = watermark.load_manifest()
    if not include_superseded:
        rows = [r for r in rows if not r.get("superseded")]
    if contains:
        needle = contains.lower()
        rows = [r for r in rows if needle in r.get("output", "").lower() or needle in r.get("source", "").lower()]
    rows.sort(key=lambda r: r.get("created", ""), reverse=True)
    keep = ("id", "output", "source", "created", "width", "height", "superseded", "superseded_by")
    return {
        "total": len(rows),
        "rows": [{k: r[k] for k in keep if k in r} for r in rows[:limit]],
    }


if __name__ == "__main__":
    mcp.run(show_banner=False)
