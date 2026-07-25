#!/usr/bin/env python3
"""Generate HTML/PDF for all NanoServe docs. Prefer: python3 scripts/generate_reports.py"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
subprocess.check_call([sys.executable, str(ROOT / "scripts" / "generate_reports.py")])
