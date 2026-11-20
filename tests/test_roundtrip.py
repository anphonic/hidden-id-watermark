"""Tests: embed IDs, find them again after common edits, and keep state safe.

Run from the repo root:  python -m unittest discover -s tests -v

Every test uses a throwaway WATERMARK_DIR, so it never touches a real key or manifest.
"""
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

SCRIPT = Path(__file__).resolve().parents[1] / "watermark.py"


def load_module():
    spec = importlib.util.spec_from_file_location("watermark_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_test_image(path, w=800, h=600, seed=1):
    """A photo-like picture: smooth gradients, shapes and some grain."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 128 + 90 * np.sin(x / 70) * np.cos(y / 90)
    g = 128 + 80 * np.cos((x + y) / 110)
    b = 128 + 70 * np.sin(np.hypot(x - w / 2, y - h / 2) / 40)
    img = np.stack([r, g, b], axis=-1) + rng.normal(0, 8, (h, w, 3))
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").save(path)


def make_other_scene(path, w=800, h=600, seed=7):
    """A genuinely different picture: random flat shapes on a dark ground."""
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (w, h), (35, 40, 50))
    d = ImageDraw.Draw(im)
    for _ in range(40):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = x0 + int(rng.integers(20, 200)), y0 + int(rng.integers(20, 200))
        colour = tuple(int(v) for v in rng.integers(0, 256, 3))
        (d.ellipse if rng.random() < 0.5 else d.rectangle)((x0, y0, x1, y1), fill=colour)
    arr = np.asarray(im, dtype=np.float32) + rng.normal(0, 4, (h, w, 3))
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").save(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ToolCase(unittest.TestCase):
    """Fresh state folder per class; runs the script as a subprocess."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.state = cls.root / "state"
        cls.state.mkdir()
        (cls.root / "originals").mkdir()
        cls.out = cls.root / "published"
        cls.env = dict(os.environ, WATERMARK_DIR=str(cls.state))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def run_tool(cls, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], env=cls.env,
                              capture_output=True, text=True, encoding="utf-8")

    @classmethod
    def embed(cls, *inputs, extra=()):
        r = cls.run_tool("embed", *inputs, "--out", cls.out, "--owner", "Test Owner", *extra)
        assert r.returncode == 0, r.stdout + r.stderr
        return r

    def manifest(self):
        return json.loads((self.state / "manifest.json").read_text(encoding="utf-8"))

    def check(self, *paths):
        r = self.run_tool("check", *paths)
        return r.returncode, r.stdout + r.stderr

    def assertMatch(self, path, hex_id, how=None):
        code, out = self.check(path)
        self.assertEqual(code, 0, out)
        self.assertIn(f"  MATCH: ", out)
        self.assertIn(f"agree with id {hex_id}", out)
        self.assertNotIn("Traceback", out)
        if how:
            self.assertIn(how, out)
        return out


class RoundTrip(ToolCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.original = cls.root / "originals" / "picture.png"
        make_test_image(cls.original)
        cls.embed(cls.original)
        cls.published = cls.out / "picture.png"
        rows = json.loads((cls.state / "manifest.json").read_text(encoding="utf-8"))
        cls.id = rows[0]["id"]

    def test_key_and_manifest_written_to_watermark_dir(self):
        self.assertTrue((self.state / "watermark.key").exists())
        self.assertEqual(len(self.manifest()), 1)

    def test_no_temp_files_left(self):
        self.assertEqual([p.name for p in self.out.iterdir()], ["picture.png"])

    def test_published_copy_matches(self):
        out = self.assertMatch(self.published, self.id, "as-is")
        self.assertIn("Chance bound", out)

    def test_resized_jpeg_matches(self):
        p = self.root / "small.jpg"
        Image.open(self.published).resize((600, 450), Image.LANCZOS).save(p, quality=80)
        self.assertMatch(p, self.id, "scaled to 800x600")

    def test_mirrored_matches(self):
        p = self.root / "mirror.png"
        ImageOps.mirror(Image.open(self.published)).save(p)
        self.assertMatch(p, self.id, "mirrored")

    def test_brightness_edit_matches(self):
        p = self.root / "bright.png"
        ImageEnhance.Brightness(Image.open(self.published).convert("RGB")).enhance(1.15).save(p)
        self.assertMatch(p, self.id)

    def test_crop_matches(self):
        p = self.root / "crop.png"
        Image.open(self.published).crop((80, 60, 720, 540)).save(p)
        self.assertMatch(p, self.id, "re-aligned as a crop")

    def test_off_grid_crop_with_other_aspect_matches(self):
        p = self.root / "crop_odd.png"
        Image.open(self.published).crop((137, 41, 611, 509)).save(p)
        self.assertMatch(p, self.id, "re-aligned as a crop")

    def test_crop_plus_tone_edit_matches(self):
        p = self.root / "crop_tone.png"
        im = Image.open(self.published).convert("RGB").crop((100, 90, 700, 540))
        im = ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(0.85)).enhance(1.1)
        im.save(p)
        self.assertMatch(p, self.id, "re-aligned as a crop")

    def test_unrelated_image_does_not_match(self):
        p = self.root / "other.png"
        make_other_scene(p)
        code, out = self.check(p)
        self.assertEqual(code, 1, out)
        self.assertIn("NO MATCH", out)

    def test_original_is_untouched_and_out_cannot_be_source_folder(self):
        before = sha(self.original)
        r = self.run_tool("embed", self.original, "--out", self.original.parent, "--owner", "X")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Refusing", r.stdout + r.stderr)
        self.assertEqual(sha(self.original), before)


class ForceReplace(ToolCase):
    def test_force_keeps_old_id_and_old_copies_still_match(self):
        src = self.root / "originals" / "scene.png"
        make_test_image(src, seed=3)
        self.embed(src)
        dst = self.out / "scene.png"
        old_copy = self.root / "old_copy.jpg"
        Image.open(dst).convert("RGB").save(old_copy, quality=88)
        old_id = self.manifest()[0]["id"]

        r = self.embed(src, extra=["--force"])
        self.assertIn("previous copy kept", r.stdout)
        rows = self.manifest()
        self.assertEqual(len(rows), 2)
        old, new = rows
        self.assertEqual(old["id"], old_id)
        self.assertEqual(old["superseded_by"], new["id"])
        self.assertTrue(Path(old["output"]).exists())
        self.assertEqual(Path(old["output"]).parent, self.state / "archive")
        self.assertEqual(sha(old["output"]), old["output_sha256"])

        out = self.assertMatch(old_copy, old_id)
        self.assertIn("Superseded", out)
        self.assertMatch(dst, new["id"], "as-is")


class StaleReference(ToolCase):
    def test_missing_and_wrong_size_references_are_skipped(self):
        a, b = self.root / "originals" / "a.png", self.root / "originals" / "b.png"
        make_test_image(a, seed=4)
        make_test_image(b, seed=5)
        self.embed(a, b)
        crop = self.root / "crop_a.png"
        Image.open(self.out / "a.png").crop((60, 50, 700, 530)).save(crop)
        (self.out / "b.png").unlink()                        # missing reference
        rows = self.manifest()
        rows.append(dict(rows[0], id="0123456789abcdef", width=999, height=777))   # wrong size
        (self.state / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")

        out = self.assertMatch(crop, rows[0]["id"], "re-aligned as a crop")
        self.assertIn("published copy missing", out)
        self.assertIn("skipped for alignment", out)


class Transparency(ToolCase):
    def test_rgba_png_keeps_alpha(self):
        src = self.root / "originals" / "logo.png"
        make_test_image(src, seed=6)
        im = Image.open(src).convert("RGBA")
        w, h = im.size
        alpha = np.full((h, w), 255, np.uint8)
        alpha[:, : w // 4] = np.linspace(0, 255, w // 4, dtype=np.uint8)   # soft left edge
        alpha[h // 2: h // 2 + 40, w // 2: w // 2 + 40] = 0                  # a hole
        im.putalpha(Image.fromarray(alpha, "L"))
        im.save(src)

        self.embed(src, extra=["--no-visible"])
        out = Image.open(self.out / "logo.png")
        self.assertEqual(out.mode, "RGBA")
        diff = np.abs(np.asarray(out.getchannel("A"), np.int16) - alpha.astype(np.int16)).max()
        self.assertLessEqual(diff, 1)

    def test_rgb_with_trns_gains_alpha(self):
        src = self.root / "originals" / "keyed.png"
        make_test_image(src, seed=8)
        im = Image.open(src).convert("RGB")
        ImageDraw.Draw(im).rectangle((300, 200, 380, 280), fill=(0, 255, 0))
        im.save(src, transparency=(0, 255, 0))

        self.embed(src, extra=["--no-visible"])
        out = Image.open(self.out / "keyed.png")
        self.assertEqual(out.mode, "RGBA")
        self.assertEqual(out.getchannel("A").getpixel((340, 240)), 0)
        self.assertEqual(out.getchannel("A").getpixel((50, 50)), 255)


class Concurrency(ToolCase):
    def test_parallel_embeds_on_fresh_state_keep_every_row_and_one_key(self):
        srcs = []
        for i in range(4):
            p = self.root / "originals" / f"p{i}.png"
            make_test_image(p, seed=20 + i)
            srcs.append(p)
        procs = [subprocess.Popen([sys.executable, str(SCRIPT), "embed", str(p), "--out", str(self.out),
                                   "--owner", "T", "--no-visible"], env=self.env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                 for p in srcs]
        logs = [p.communicate()[0] for p in procs]
        for p, log in zip(procs, logs):
            self.assertEqual(p.returncode, 0, log)
        self.assertEqual(sum("Created new secret key" in log for log in logs), 1, logs)
        rows = self.manifest()
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({r["id"] for r in rows}), 4)
        code, out = self.check(*(self.out / f"p{i}.png" for i in range(4)))
        self.assertEqual(code, 0, out)
        self.assertEqual(out.count("  MATCH: "), 4, out)


class InProcess(unittest.TestCase):
    """Direct calls into the module, with HERE pointed at a temp state folder."""

    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.mod.HERE = self.root / "state"
        self.mod.HERE.mkdir()
        (self.root / "originals").mkdir()
        self.out = self.root / "published"

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = self.mod.main([str(a) for a in argv])
        return code, buf.getvalue()

    def test_failed_readback_with_force_keeps_old_copy_and_manifest(self):
        src = self.root / "originals" / "pic.png"
        make_test_image(src, seed=11)
        code, log = self.run_main("embed", src, "--out", self.out, "--owner", "T")
        self.assertEqual(code, 0, log)
        dst = self.out / "pic.png"
        before_file, before_manifest = sha(dst), (self.mod.HERE / "manifest.json").read_text()

        with mock.patch.object(self.mod, "extract_bits", return_value=[False] * 64):
            code, log = self.run_main("embed", src, "--out", self.out, "--owner", "T", "--force")
        self.assertEqual(code, 1, log)
        self.assertIn("FAIL", log)
        self.assertIn("left as it was", log)
        self.assertEqual(sha(dst), before_file)
        self.assertEqual((self.mod.HERE / "manifest.json").read_text(), before_manifest)
        self.assertEqual([p.name for p in self.out.iterdir()], ["pic.png"])
        self.assertFalse((self.mod.HERE / "archive").exists())

    def test_panorama_embeds(self):
        src = self.root / "originals" / "pano.png"
        make_test_image(src, w=6000, h=300, seed=12)
        code, log = self.run_main("embed", src, "--out", self.out, "--owner", "T")
        self.assertEqual(code, 0, log)
        with Image.open(self.out / "pano.png") as im:
            self.assertEqual(im.size, (6000, 300))

    def test_visible_mark_covers_the_image(self):
        im = Image.new("RGB", (900, 400), (0, 0, 0))
        marked = np.asarray(self.mod.add_visible(im, "© Test", 0.5))
        lit = marked.max(axis=2) > 0
        for part in (lit[:, :300], lit[:, 300:600], lit[:, 600:]):
            self.assertGreater(part.mean(), 0.01)

    def test_flat_suspect_is_not_aligned(self):
        published = Image.open(self._picture()).convert("RGB")
        flat = Image.new("RGB", (400, 300), (128, 128, 128))
        restored, _score, box = self.mod.realign_crop(published, flat)
        self.assertIsNone(restored)
        self.assertIsNone(box)

    def test_verdict_accounts_for_comparisons(self):
        m = self.mod
        self.assertEqual(m.verdict(52, m.chance_of(52, 100_000)), "NO MATCH")
        self.assertEqual(m.verdict(58, m.chance_of(58, 12)), "MATCH")
        self.assertEqual(m.verdict(55, m.chance_of(55, 12)), "LIKELY MATCH")
        self.assertEqual(m.verdict(40, m.chance_of(40, 1)), "NO MATCH")
        self.assertEqual(m.chance_of(64, 0), m.chance_of(64, 1))

    def test_tone_match_rejects_shape_mismatch(self):
        a = Image.new("RGB", (100, 80))
        b = Image.new("RGB", (80, 100))
        self.assertIsNone(self.mod.tone_match(a, b))

    def _embedded(self, name="pic.png", seed=11):
        src = self.root / "originals" / name
        make_test_image(src, seed=seed)
        code, log = self.run_main("embed", src, "--out", self.out, "--owner", "T", "--no-visible")
        self.assertEqual(code, 0, log)
        return src, self.out / name

    def _archive_files(self):
        a = self.mod.HERE / "archive"
        return sorted(p.name for p in a.iterdir()) if a.exists() else []

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_published_copy_gets_normal_permissions(self):
        old = os.umask(0o022)
        try:
            src, dst = self._embedded()
            self.assertEqual(dst.stat().st_mode & 0o777, 0o644)
            os.chmod(dst, 0o640)
            code, log = self.run_main("embed", src, "--out", self.out, "--owner", "T", "--no-visible", "--force")
            self.assertEqual(code, 0, log)
            self.assertEqual(dst.stat().st_mode & 0o777, 0o640)
        finally:
            os.umask(old)

    def test_key_not_written_when_it_cannot_be_restricted(self):
        src = self.root / "originals" / "pic.png"
        make_test_image(src, seed=11)
        with mock.patch.object(self.mod, "restrict_to_owner", side_effect=OSError("denied")):
            with self.assertRaises(SystemExit) as cm:
                self.run_main("embed", src, "--out", self.out, "--owner", "T")
        self.assertIn("No key was written", str(cm.exception.code))
        self.assertEqual([p.name for p in self.mod.HERE.iterdir() if p.name != ".watermark.lock"], [])

    def test_key_is_restricted_while_still_empty(self):
        sizes = []
        real = self.mod.restrict_to_owner

        def spy(path):
            sizes.append(os.path.getsize(path))
            real(path)
        with mock.patch.object(self.mod, "restrict_to_owner", side_effect=spy):
            self.mod.load_key(create=True)
        self.assertEqual(sizes, [0])
        self.assertTrue(self.mod.key_file().exists())

    def test_archive_names_never_collide(self):
        a, b = self.root / "a", self.root / "b"
        a.mkdir()
        b.mkdir()
        (a / "x.png").write_bytes(b"first")
        (b / "x.png").write_bytes(b"second")
        with mock.patch.object(self.mod, "now_utc", return_value="t"):
            t1, _ = self.mod.archive_replaced(a / "x.png", [], "1" * 16)
            t2, _ = self.mod.archive_replaced(b / "x.png", [], "2" * 16)
            t3, _ = self.mod.archive_replaced(a / "x.png", [], "3" * 16)
        self.assertEqual(len({t1, t2, t3}), 3)
        self.assertEqual(t1.read_bytes(), b"first")
        self.assertEqual(t2.read_bytes(), b"second")

    def test_tracked_archive_name_is_not_reused(self):
        dst = self.root / "x.png"
        dst.write_bytes(b"new")
        rows = [{"id": "ab" * 8, "output": str(dst)}]
        (self.mod.HERE / "archive").mkdir()
        existing = self.mod.HERE / "archive" / ("ab" * 8 + ".png")
        existing.write_bytes(b"old")
        target, updated = self.mod.archive_replaced(dst, rows, "cd" * 8)
        self.assertNotEqual(target, existing)
        self.assertEqual(existing.read_bytes(), b"old")
        self.assertNotIn("superseded", rows[0])            # caller's list is untouched
        self.assertEqual(updated[0]["output"], str(target))

    def test_manifest_save_failure_leaves_old_copy_in_place(self):
        src, dst = self._embedded()
        before_file, before_manifest = sha(dst), (self.mod.HERE / "manifest.json").read_text()
        with mock.patch.object(self.mod, "save_manifest", side_effect=OSError("disk full")):
            code, log = self.run_main("embed", src, "--out", self.out, "--owner", "T", "--no-visible", "--force")
        self.assertEqual(code, 1, log)
        self.assertIn("could not record it", log)
        self.assertEqual(sha(dst), before_file)
        self.assertEqual((self.mod.HERE / "manifest.json").read_text(), before_manifest)
        self.assertEqual(self._archive_files(), [])
        self.assertEqual([p.name for p in self.out.iterdir()], ["pic.png"])

    def test_replace_failure_rolls_back_manifest(self):
        src, dst = self._embedded()
        before_file, before_manifest = sha(dst), (self.mod.HERE / "manifest.json").read_text()
        real = self.mod.replace_file

        def failing(a, b):
            if Path(b) == dst:
                raise PermissionError("locked")
            real(a, b)
        with mock.patch.object(self.mod, "replace_file", side_effect=failing):
            code, log = self.run_main("embed", src, "--out", self.out, "--owner", "T", "--no-visible", "--force")
        self.assertEqual(code, 1, log)
        self.assertIn("It was left as it was", log)
        self.assertEqual(sha(dst), before_file)
        self.assertEqual(json.loads((self.mod.HERE / "manifest.json").read_text()), json.loads(before_manifest))
        self.assertEqual(self._archive_files(), [])

    def test_reference_cache_is_bounded(self):
        rows = []
        for i in range(6):
            p = self.root / f"ref{i}.png"
            make_test_image(p, w=300, h=260, seed=30 + i)
            rows.append({"output": str(p), "width": 300, "height": 260, "output_sha256": sha(p)})
        cache = self.mod.ReferenceCache(limit=2)
        for _ in range(2):
            for r in rows:
                img, note = cache.get(r)
                self.assertIsNotNone(img)
                self.assertIsNone(note)
                self.assertLessEqual(len(cache.images), 2)

    def test_reference_cache_checks_each_row_on_its_own(self):
        p = self.root / "ref.png"
        make_test_image(p, w=300, h=260, seed=40)
        good = {"output": str(p), "width": 300, "height": 260, "output_sha256": sha(p)}
        wrong = dict(good, width=999, height=777)
        for order in ((wrong, good), (good, wrong)):
            cache = self.mod.ReferenceCache()
            results = {id(r): cache.get(r) for r in order}
            self.assertIsNotNone(results[id(good)][0])
            self.assertIsNone(results[id(wrong)][0])
            self.assertIn("skipped for alignment", results[id(wrong)][1])

    def test_search_gates_use_full_match_rule(self):
        self.assertTrue(self.mod.is_match(58, 1))
        self.assertFalse(self.mod.is_match(58, 10 ** 6))   # enough bits, chance bound too big
        self.assertTrue(self.mod.is_match(64, 10 ** 6))

    def _picture(self):
        p = self.root / "originals" / "ref.png"
        make_test_image(p, seed=13)
        return p


if __name__ == "__main__":
    unittest.main()
