"""End-to-end test: embed an ID, then find it again after common edits.

Run from the repo root:  python -m unittest discover -s tests -v

Uses a throwaway WATERMARK_DIR, so it never touches a real key or manifest.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

SCRIPT = Path(__file__).resolve().parents[1] / "watermark.py"


def make_test_image(path, w=800, h=600, seed=1):
    """A photo-like picture: smooth gradients, shapes and some grain."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 128 + 90 * np.sin(x / 70) * np.cos(y / 90)
    g = 128 + 80 * np.cos((x + y) / 110)
    b = 128 + 70 * np.sin(np.hypot(x - w / 2, y - h / 2) / 40)
    img = np.stack([r, g, b], axis=-1) + rng.normal(0, 8, (h, w, 3))
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").save(path)


class RoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.env = dict(os.environ, WATERMARK_DIR=str(root / "state"))
        (root / "state").mkdir()
        (root / "originals").mkdir()
        cls.original = root / "originals" / "picture.png"
        make_test_image(cls.original)
        cls.out = root / "published"
        r = cls.run_tool("embed", str(cls.original), "--out", str(cls.out), "--owner", "Test Owner")
        assert r.returncode == 0, r.stdout + r.stderr
        cls.published = cls.out / "picture.png"
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def run_tool(cls, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args], env=cls.env,
                              capture_output=True, text=True, encoding="utf-8")

    def check(self, image_path):
        r = self.run_tool("check", str(image_path))
        return r.returncode, r.stdout + r.stderr

    def test_key_and_manifest_written_to_watermark_dir(self):
        state = Path(self.env["WATERMARK_DIR"])
        self.assertTrue((state / "watermark.key").exists())
        self.assertTrue((state / "manifest.json").exists())

    def test_published_copy_matches(self):
        code, out = self.check(self.published)
        self.assertEqual(code, 0, out)
        self.assertIn("MATCH", out)

    def test_resized_jpeg_matches(self):
        p = self.root / "small.jpg"
        Image.open(self.published).resize((600, 450), Image.LANCZOS).save(p, quality=80)
        code, out = self.check(p)
        self.assertEqual(code, 0, out)

    def test_mirrored_matches(self):
        p = self.root / "mirror.png"
        ImageOps.mirror(Image.open(self.published)).save(p)
        code, out = self.check(p)
        self.assertEqual(code, 0, out)

    def test_brightness_edit_matches(self):
        p = self.root / "bright.png"
        ImageEnhance.Brightness(Image.open(self.published).convert("RGB")).enhance(1.15).save(p)
        code, out = self.check(p)
        self.assertEqual(code, 0, out)

    def test_crop_matches(self):
        p = self.root / "crop.png"
        Image.open(self.published).crop((80, 60, 720, 540)).save(p)
        code, out = self.check(p)
        self.assertEqual(code, 0, out)

    def test_unrelated_image_does_not_match(self):
        p = self.root / "other.png"
        make_test_image(p, seed=99)
        code, out = self.check(p)
        self.assertEqual(code, 1, out)
        self.assertIn("NO MATCH", out)

    def test_original_is_untouched_and_out_cannot_be_source_folder(self):
        r = self.run_tool("embed", str(self.original), "--out", str(self.original.parent), "--owner", "X")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Refusing", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
