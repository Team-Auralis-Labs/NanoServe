#!/usr/bin/env python3
"""WASM bundle structure and optional Emscripten build smoke tests."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestWasmBundle(unittest.TestCase):
    def test_static_files_present(self):
        wasm_dir = ROOT / "deployment/wasm"
        for name in ("index.html", "app.js", "nanoserve.js", "styles.css", "build.sh"):
            p = wasm_dir / name
            self.assertTrue(p.exists(), f"missing {p}")

    def test_build_scripts_executable(self):
        for rel in ("scripts/build_wasm.sh", "deployment/wasm/build.sh"):
            p = ROOT / rel
            self.assertTrue(p.exists())
            self.assertTrue(os.access(p, os.X_OK), f"{rel} not executable")

    def test_package_json_scripts(self):
        pkg = ROOT / "package.json"
        self.assertTrue(pkg.exists())
        text = pkg.read_text()
        self.assertIn("build:wasm", text)
        self.assertIn("serve:wasm", text)

    def test_nanoq_load_buffer_native(self):
        if not (ROOT / "engine/build/libnanoserve_engine.so").exists():
            self.skipTest("Engine not built — run cd engine/build && cmake .. && make")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tests/test_wasm_native.py")],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    @unittest.skipUnless(shutil.which("emcc"), "Emscripten not installed")
    def test_emscripten_build(self):
        env = os.environ.copy()
        proc = subprocess.run(
            [str(ROOT / "scripts/build_wasm.sh")],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertTrue((ROOT / "deployment/wasm/nanoserve_engine.wasm").exists())
        self.assertTrue((ROOT / "deployment/wasm/nanoserve_engine.js").exists())


if __name__ == "__main__":
    unittest.main()
