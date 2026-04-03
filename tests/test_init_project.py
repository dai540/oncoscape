from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from oncoscape.bootstrap import initialize_hpc_project


class TestInitProject(unittest.TestCase):
    def test_initialize_hpc_project_writes_config_and_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            code_root = Path(__file__).resolve().parents[1]
            result = initialize_hpc_project(project_root=root / "project", code_root=code_root, dry_run=False)
            generated = Path(result["generated_config"])
            self.assertTrue(generated.exists())
            payload = yaml.safe_load(generated.read_text(encoding="utf-8"))
            self.assertEqual(payload["paths"]["code_root"], str(code_root.resolve()))
            self.assertTrue(Path(result["downloads_config"]).exists())
            self.assertTrue(Path(result["sources_template"]).exists())


if __name__ == "__main__":
    unittest.main()
