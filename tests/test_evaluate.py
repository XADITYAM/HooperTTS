import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluate import evaluate_dataset


def test_evaluate_dataset_writes_reports() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        dataset_dir = root / "dataset"
        nested_dir = dataset_dir / "nested"
        output_dir = root / "reports"
        nested_dir.mkdir(parents=True)
        (dataset_dir / "one.txt").write_text(
            "Imagine HooperTTS. Officially confirmed.", encoding="utf-8"
        )
        (nested_dir / "two.txt").write_text(
            "But the NPC noticed everything. Subscribe for more.", encoding="utf-8"
        )

        results = evaluate_dataset(dataset_dir, output_dir)

        assert len(results) == 2
        assert (output_dir / "results.csv").exists()
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "summary.md").exists()

        with (output_dir / "results.csv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))

        assert {row["Script Name"] for row in rows} == {"one.txt", "nested/two.txt"}
        assert rows[0]["Profile"] == "default"
        assert "Narration Score" in rows[0]

        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["script_count"] == 2
        assert summary["profile_distribution"] == {"default": 2}
        assert "optimization_totals" in summary

        summary_md = (output_dir / "summary.md").read_text(encoding="utf-8")
        assert "Average Statistics" in summary_md
        assert "Optimization Totals" in summary_md


if __name__ == "__main__":
    test_evaluate_dataset_writes_reports()
