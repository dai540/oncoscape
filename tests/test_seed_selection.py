from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from oncoscape.reporting.selection import compute_selection_score, summarize_seed_sweep


class TestSeedSelection(unittest.TestCase):
    def test_compute_selection_score_prefers_higher_correlations_and_lower_js(self) -> None:
        weak = {
            "compartment_macro_f1": 0.0,
            "composition_mean_pearson": 0.05,
            "program_mean_pearson": 0.01,
            "composition_js_divergence": 0.10,
        }
        strong = {
            "compartment_macro_f1": 0.0,
            "composition_mean_pearson": 0.08,
            "program_mean_pearson": 0.04,
            "composition_js_divergence": 0.02,
        }
        self.assertGreater(compute_selection_score(strong), compute_selection_score(weak))

    def test_summarize_seed_sweep_promotes_best_validation_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed_root = root / "seed_sweep"
            report_dir = root / "reports"
            checkpoint_dir = root / "checkpoints"
            predictions_dir = root / "predictions"
            for seed, comp, jsd, prog in [("3", 0.01, 0.02, 0.00), ("7", 0.04, 0.01, 0.03)]:
                seed_dir = seed_root / f"seed_{seed}"
                (seed_dir / "reports").mkdir(parents=True)
                (seed_dir / "checkpoints").mkdir(parents=True)
                (seed_dir / "predictions" / f"slide_{seed}").mkdir(parents=True)
                pd.DataFrame(
                    [
                        {
                            "epoch": 15,
                            "macro_f1": 0.0,
                            "balanced_accuracy": 0.0,
                            "composition_mean_pearson": comp,
                            "composition_js_divergence": jsd,
                            "program_mean_pearson": prog,
                        }
                    ]
                ).to_csv(seed_dir / "reports" / "val_metrics.csv", index=False)
                (seed_dir / "reports" / "test_metrics.json").write_text(
                    json.dumps({"composition_mean_pearson": comp + 0.01, "program_mean_pearson": prog}),
                    encoding="utf-8",
                )
                (seed_dir / "checkpoints" / "best.pt").write_text(f"seed-{seed}", encoding="utf-8")
                (seed_dir / "predictions" / f"slide_{seed}" / "tile_predictions.parquet").write_text("ok", encoding="utf-8")

            config = {
                "selection": {
                    "seed_sweep_dir": str(seed_root),
                    "summary_json": str(report_dir / "seed_sweep_summary.json"),
                    "summary_csv": str(report_dir / "seed_sweep_summary.csv"),
                    "promote_best": True,
                },
                "train_run": {"checkpoint_dir": str(checkpoint_dir)},
                "render": {"predictions_dir": str(predictions_dir), "report_dir": str(report_dir)},
            }
            result = summarize_seed_sweep(config, dry_run=False)
            self.assertEqual(result["best_seed"], "7")
            self.assertTrue((checkpoint_dir / "best.pt").exists())
            self.assertTrue((report_dir / "seed_sweep_summary.json").exists())
            payload = json.loads((report_dir / "seed_sweep_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["best_seed"], "7")


if __name__ == "__main__":
    unittest.main()
