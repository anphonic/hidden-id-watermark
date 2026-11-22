---
name: check-image
description: Check whether a suspect image is a copy of one of the user's watermarked images. Use when the user asks if an image is theirs, was stolen, came from their shop or site, or wants a watermark check.
argument-hint: <image path> [more paths]
---

Check the suspect images with the `check_image` tool from the hidden-id-watermark MCP server.

1. Use absolute paths. If the user gave a URL or an image in chat rather than a file, ask them to save it to disk first. Don't download it yourself.
2. Leave `crop_search` on unless the user asks for a quick check. It's what finds cropped or colour-edited copies, and it can take several minutes with a large manifest. Tell the user it may take a while before you start.
3. Report each image's verdict exactly as the report gives it:
   - **MATCH** and **LIKELY MATCH** are different verdicts. Don't upgrade one to the other.
   - Name the published copy and the original it points to, and the date it was watermarked.
   - The chance bound limits how likely an unrelated image is to score that well. It is not the probability that the attribution is wrong, so don't describe it that way.
4. A NO MATCH means the ID wasn't found, not that the image isn't theirs: heavy edits, rotation or AI regeneration destroy it. Say so.
5. If `errors` is not empty, show it. A missing key or empty manifest means nothing has been watermarked from this install yet.

Never open, print or copy `watermark.key`.

Images to check: $ARGUMENTS
