"""Tests for the MCP server: drive it through FastMCP's in-memory client.

Run from the repo root:  python -m unittest discover -s tests -v
Skipped unless fastmcp is installed (pip install -r requirements-mcp.txt).

Every test uses a throwaway WATERMARK_DIR, so it never touches a real key or manifest.
"""
import asyncio
import importlib.util
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_roundtrip import make_other_scene, make_test_image  # noqa: E402

try:
    from fastmcp import Client
except ImportError:  # pragma: no cover
    Client = None

SERVER = Path(__file__).resolve().parents[1] / "mcp_server.py"


def load_server():
    spec = importlib.util.spec_from_file_location("mcp_server_under_test", SERVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipIf(Client is None, "fastmcp not installed")
class McpServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.state = cls.root / "state"
        cls.state.mkdir()
        cls.originals = cls.root / "originals"
        cls.originals.mkdir()
        cls.out = cls.root / "published"
        make_test_image(cls.originals / "picture.png")
        make_other_scene(cls.root / "unrelated.png")
        cls.env = mock.patch.dict(os.environ, {"WATERMARK_DIR": str(cls.state)})
        cls.env.start()
        cls.server = load_server()

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        cls.tmp.cleanup()

    def call(self, tool, args, **client_opts):
        async def go():
            async with Client(self.server.mcp, **client_opts) as client:
                return await client.call_tool(tool, args, raise_on_error=False)
        return asyncio.run(go())

    def embed(self, **extra):
        args = {"inputs": [str(self.originals)], "out": str(self.out), "owner": "Test Studio", **extra}
        return self.call("watermark_images", args)

    def test_1_check_before_anything_is_watermarked(self):
        res = self.call("check_image", {"paths": [str(self.root / "unrelated.png")]}).structured_content
        self.assertFalse(res["matched"])
        self.assertIn("No key file", res["errors"])

    def test_2_tools_are_listed_with_hints(self):
        async def go():
            async with Client(self.server.mcp) as client:
                return {t.name: t for t in await client.list_tools()}
        tools = asyncio.run(go())
        self.assertEqual(set(tools), {"check_image", "watermark_images", "list_watermarked"})
        self.assertTrue(tools["check_image"].annotations.read_only_hint)
        self.assertTrue(tools["list_watermarked"].annotations.read_only_hint)
        self.assertFalse(tools["watermark_images"].annotations.read_only_hint)

    def test_3_embed_then_list_then_check(self):
        res = self.embed().structured_content
        self.assertTrue(res["ok"], res)
        published = self.out / "picture.png"
        self.assertTrue(published.exists())

        rows = self.call("list_watermarked", {}).structured_content
        self.assertEqual(rows["total"], 1)
        self.assertEqual(rows["rows"][0]["output"], str(published))
        self.assertNotIn("output_sha256", rows["rows"][0])

        res = self.call("check_image", {"paths": [str(published)], "crop_search": False}).structured_content
        self.assertTrue(res["matched"], res)
        self.assertIn("MATCH", res["report"])
        self.assertIn(rows["rows"][0]["id"], res["report"])

        res = self.call("check_image", {"paths": [str(self.root / "unrelated.png")],
                                        "crop_search": False}).structured_content
        self.assertFalse(res["matched"])
        self.assertIn("NO MATCH", res["report"])

    def test_4_embed_skips_existing_copy_without_force(self):
        before = (self.out / "picture.png").read_bytes()
        res = self.embed().structured_content
        self.assertTrue(res["ok"])
        self.assertIn("skip", res["report"].lower())
        self.assertEqual((self.out / "picture.png").read_bytes(), before)

    def test_5_list_filters(self):
        self.assertEqual(self.call("list_watermarked", {"contains": "PICTURE"}).structured_content["total"], 1)
        self.assertEqual(self.call("list_watermarked", {"contains": "nothing-like-this"}).structured_content["total"], 0)

    def test_6_bad_arguments_are_rejected(self):
        self.assertTrue(self.call("check_image", {"paths": []}).is_error)
        self.assertTrue(self.embed(opacity=2).is_error)

    def test_7_long_runs_report_progress(self):
        seen = []

        async def on_progress(progress, total, message):
            seen.append(message)

        with mock.patch.object(self.server, "HEARTBEAT_SECONDS", 0.05):
            self.call("check_image", {"paths": [str(self.out / "picture.png")]}, progress_handler=on_progress)
        self.assertTrue(any("still working" in (m or "") for m in seen), seen)


class Paths(unittest.TestCase):

    @unittest.skipIf(Client is None, "fastmcp not installed")
    def test_option_like_names_become_absolute_paths(self):
        server = load_server()
        for p in server.absolute(["-x.png", "--force"]):
            self.assertTrue(Path(p).is_absolute())
            self.assertFalse(p.startswith("-"))


@unittest.skipIf(Client is None or os.name == "nt", "needs fastmcp and POSIX signals")
class Cancelling(unittest.TestCase):
    """A cancelled check stops its child process; a cancelled embed is left to finish."""

    def run_and_cancel(self, kill_on_cancel):
        server = load_server()
        with tempfile.TemporaryDirectory() as d:
            pid_file = Path(d) / "pid"
            fake = Path(d) / "fake.py"
            fake.write_text(f"import os, time\nopen({str(pid_file)!r}, 'w').write(str(os.getpid()))\ntime.sleep(60)\n")

            class Ctx:
                async def report_progress(self, *a):
                    pass

            async def go():
                task = asyncio.ensure_future(server.run_watermark([], Ctx(), kill_on_cancel=kill_on_cancel))
                for _ in range(200):
                    if pid_file.exists() and pid_file.read_text():
                        break
                    await asyncio.sleep(0.05)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                return int(pid_file.read_text())

            with mock.patch.object(server, "SCRIPT", fake):
                pid = asyncio.run(go())
            return pid

    @staticmethod
    def alive(pid):
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    def test_cancelled_check_is_killed(self):
        pid = self.run_and_cancel(kill_on_cancel=True)
        for _ in range(40):
            if not self.alive(pid):
                break
            time.sleep(0.05)
        self.assertFalse(self.alive(pid))

    def test_cancelled_embed_keeps_running(self):
        pid = self.run_and_cancel(kill_on_cancel=False)
        try:
            self.assertTrue(self.alive(pid))
        finally:
            os.kill(pid, 9)


if __name__ == "__main__":
    unittest.main()
