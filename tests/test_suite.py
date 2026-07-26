#!/usr/bin/env python3
"""NanoServe simulated integration test suite."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LD_LIBRARY_PATH", str(ROOT / "allocator" / "target" / "release"))
os.environ.setdefault("NANOSERVE_ENGINE_LIB", str(ROOT / "engine" / "build" / "libnanoserve_engine.so"))

from nanoserve import NanoServe, Quantizer
from nanoserve.engine.pool import EnginePool
from nanoserve.engine.worker import BackendKind, EngineWorker


class TestQuantizer(unittest.TestCase):
    def test_quantize_int8_shape(self):
        w = np.random.randn(64, 128).astype(np.float32)
        q, scales = Quantizer.quantize_int8(w)
        self.assertEqual(q.shape, w.shape)
        self.assertEqual(scales.shape, (64, 1))

    def test_write_nanoq(self):
        w = np.random.randn(8, 16).astype(np.float32)
        q, scales = Quantizer.quantize_int8(w)
        out = Path("/tmp/nanoserve_test.nanoq")
        Quantizer.write_nanoq(str(out), q, scales, 8, 16)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 0)


class TestEngineFFI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = os.environ["NANOSERVE_ENGINE_LIB"]
        if not Path(cls.lib).exists():
            raise unittest.SkipTest(f"Engine lib not built: {cls.lib}")

    def test_cpu_infer_deterministic(self):
        w = EngineWorker(lib_path=self.lib, backend=BackendKind.CPU)
        a = w.infer("hello", 8)
        b = w.infer("hello", 8)
        self.assertEqual(a, b)
        w.cleanup()

    def test_probes(self):
        cuda = EngineWorker.probe_cuda(self.lib)
        ocl = EngineWorker.probe_opencl(self.lib)
        self.assertIsInstance(cuda, bool)
        self.assertIsInstance(ocl, bool)

    def test_cpu_cuda_output_parity(self):
        """CPU vs CUDA must produce identical token stream when CUDA is compiled + available."""
        if not EngineWorker.probe_cuda(self.lib):
            self.skipTest("CUDA not available in this build/runtime")
        cpu = EngineWorker(lib_path=self.lib, backend=BackendKind.CPU)
        gpu = EngineWorker(lib_path=self.lib, backend=BackendKind.CUDA)
        prompt = "parity check prompt"
        cpu_out = cpu.infer(prompt, 16)
        gpu_out = gpu.infer(prompt, 16)
        cpu.cleanup()
        gpu.cleanup()
        self.assertEqual(cpu_out, gpu_out, "CUDA GEMV must match CPU AVX2 for same seed/weights")


class TestEnginePool(unittest.TestCase):
    def setUp(self):
        if not Path(os.environ["NANOSERVE_ENGINE_LIB"]).exists():
            self.skipTest("Engine not built")

    def test_cpu_submit(self):
        pool = EnginePool(num_workers=2)
        r = pool.submit("test", 8, device="cpu").result()
        self.assertEqual(r.device, "cpu")
        self.assertTrue(r.text)

    def test_gpu_fallback_simulated(self):
        pool = EnginePool(num_workers=2)
        r = pool.submit("test", 8, device="gpu").result()
        if pool.gpu_available:
            self.assertIn(r.device, ("cuda", "opencl"))
        else:
            self.assertEqual(r.device, "cpu")
            self.assertTrue(any("fallback" in w.lower() or "unavailable" in w.lower() for w in r.warnings))


class TestNanoServeSDK(unittest.TestCase):
    def test_generate_devices(self):
        if not Path(os.environ["NANOSERVE_ENGINE_LIB"]).exists():
            self.skipTest("Engine not built")
        for dev in ("cpu", "gpu", "auto"):
            engine = NanoServe(device=dev)
            text = engine.generate("sdk test", max_tokens=8)
            self.assertTrue(len(text) > 0)
            self.assertIn(engine.last_device, ("cpu", "cuda", "opencl"))


class TestMemoryStress(unittest.TestCase):
    """Repeated inference to verify scratch-pool reuse (no leak/crash)."""

    @classmethod
    def setUpClass(cls):
        cls.lib = os.environ["NANOSERVE_ENGINE_LIB"]
        if not Path(cls.lib).exists():
            raise unittest.SkipTest("Engine not built")

    def test_cpu_repeated_infer(self):
        w = EngineWorker(lib_path=self.lib, backend=BackendKind.CPU)
        for i in range(200):
            text = w.infer(f"stress {i}", 12)
            self.assertTrue(text)
        w.cleanup()

    def test_cpu_cuda_parity_under_load(self):
        if not EngineWorker.probe_cuda(self.lib):
            self.skipTest("CUDA not available")
        cpu = EngineWorker(lib_path=self.lib, backend=BackendKind.CPU)
        gpu = EngineWorker(lib_path=self.lib, backend=BackendKind.CUDA)
        for i in range(50):
            p = f"load test {i}"
            self.assertEqual(cpu.infer(p, 8), gpu.infer(p, 8))
        cpu.cleanup()
        gpu.cleanup()

    def test_pool_concurrent_cpu(self):
        from concurrent.futures import ThreadPoolExecutor

        pool = EnginePool(num_workers=4)

        def one(i: int):
            return pool.submit(f"concurrent {i}", 8, device="cpu").result().text

        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(one, range(32)))
        self.assertEqual(len(results), 32)
        self.assertTrue(all(results))


class TestHTTPServer(unittest.TestCase):
    _proc: subprocess.Popen | None = None
    base = "http://127.0.0.1:8765"

    @classmethod
    def setUpClass(cls):
        if not Path(os.environ["NANOSERVE_ENGINE_LIB"]).exists():
            raise unittest.SkipTest("Engine not built")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        cls._proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8765"],
            cwd=str(ROOT / "server"),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(30):
            try:
                r = httpx.get(f"{cls.base}/health", timeout=1.0)
                if r.status_code == 200:
                    return
            except Exception:
                time.sleep(0.2)
        cls._proc.kill()
        raise RuntimeError("Server failed to start")

    @classmethod
    def tearDownClass(cls):
        if cls._proc:
            cls._proc.terminate()
            cls._proc.wait(timeout=5)

    def test_health(self):
        r = httpx.get(f"{self.base}/health")
        data = r.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("gpu_cuda", data)
        self.assertIn("models_registered", data)
        self.assertIn("gguf_available", data)
        self.assertIn("active_format", data)
        self.assertIn("native_available", data)

    def test_list_models(self):
        r = httpx.get(f"{self.base}/v1/models")
        self.assertEqual(r.status_code, 200)
        self.assertIn("models", r.json())

    def test_completions_cpu(self):
        r = httpx.post(
            f"{self.base}/v1/completions",
            json={"prompt": "hello", "max_tokens": 8, "device": "cpu"},
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["device"], "cpu")
        self.assertTrue(data["text"])

    def test_completions_gpu_or_fallback(self):
        r = httpx.post(
            f"{self.base}/v1/completions",
            json={"prompt": "hello", "max_tokens": 8, "device": "gpu"},
        )
        data = r.json()
        self.assertEqual(r.status_code, 200)
        if data["device"] == "cpu":
            self.assertTrue(len(data.get("warnings", [])) > 0)


def run_tests() -> dict:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in (
        TestQuantizer,
        TestEngineFFI,
        TestEnginePool,
        TestNanoServeSDK,
        TestMemoryStress,
        TestHTTPServer,
    ):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "passed": result.testsRun - len(result.failures) - len(result.errors),
        "success": result.wasSuccessful(),
    }


if __name__ == "__main__":
    summary = run_tests()
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    sys.exit(0 if summary["success"] else 1)
