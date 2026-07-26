#!/usr/bin/env python3
"""Download helper tests (URL path, no network for HF)."""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoserve.models.download import download_model
from nanoserve.models.registry import ModelRegistry


class TestModelsDownload(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.reg = ModelRegistry(root=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @patch("nanoserve.models.download.urllib.request.urlretrieve")
    def test_url_download(self, mock_retrieve):
        dest_dir = self.tmp / "url-model"
        dest_dir.mkdir(parents=True)
        dest = dest_dir / "weights.bin"
        dest.write_bytes(b"fake-weights")

        def fake(url, path):
            Path(path).write_bytes(b"fake-weights")

        mock_retrieve.side_effect = fake
        entry = download_model(
            source="url",
            url="https://example.com/weights.bin",
            model_id="url-model",
            registry=self.reg,
        )
        self.assertEqual(entry.id, "url-model")
        self.assertTrue(Path(entry.source_path).exists())


if __name__ == "__main__":
    unittest.main()
