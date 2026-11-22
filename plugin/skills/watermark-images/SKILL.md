---
name: watermark-images
description: Write watermarked copies of the user's images (visible mark, hidden 64-bit ID, copyright note) before they publish them. Use when the user wants to watermark, mark or protect images they are about to post or sell.
argument-hint: <images or folder> --out <folder> --owner "<name>"
disable-model-invocation: true
---

Watermark the user's images with the `watermark_images` tool from the hidden-id-watermark MCP server.

1. You need `inputs`, an `out` folder and an `owner` name. Ask only for what the user hasn't given. Use `list_watermarked` to see the owner name and folders used before, and suggest those.
2. `out` must be a separate folder from the originals. The tool refuses otherwise. The originals are never changed.
3. Set `force` only if the user said to replace existing copies. Without it, images already in `out` are skipped and listed as skipped.
4. Keep the defaults (visible mark on, opacity 0.16, quality 92, headroom 12) unless the user asks otherwise.
5. Report how many were written, skipped and failed, with any warnings, such as a transparent image whose ID is weak once flattened.
6. Remind the user once to back up `watermark.key`, `manifest.json` and `archive/`, and to keep the unwatermarked originals somewhere separate.

Never open, print or copy `watermark.key`.

Request: $ARGUMENTS
