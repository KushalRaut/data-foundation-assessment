"""Independent source, leakage and pairing checks. Run after prepare_data.py."""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent


def validate():
    for record in json.loads((ROOT / "source_hashes.json").read_text()):
        actual = hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest()
        assert actual == record["sha256"], record["path"]
    for pdf in (ROOT / "raw/fifa").glob("*.pdf"):
        header = PdfReader(pdf).pages[0].extract_text().splitlines()[:3]
        archived = pdf.with_suffix(".txt").read_text().splitlines()[:3]
        assert [" ".join(x.split()) for x in header] == [
            " ".join(x.split()) for x in archived
        ], pdf
    a = pd.read_csv(ROOT / "regression_2_1.csv")
    b = pd.read_csv(ROOT / "regression_2_2.csv")
    matches = pd.read_csv(ROOT / "matches_fifa.csv")
    meta = pd.read_csv(ROOT / "team_match_metadata.csv")
    assert a.shape == (104, 10) and b.shape == (208, 10)
    assert a.match_id.is_unique and b.row_id.is_unique
    assert (
        np.isfinite(a.iloc[:, 1:]).all().all()
        and np.isfinite(b.iloc[:, 1:]).all().all()
    )
    assert set(a.match_id) == {f"M{i:03d}" for i in range(1, 105)}
    expected_a = matches.set_index("match_id").eval("goals_a - goals_b")
    assert (
        a.set_index("match_id")
        .goal_difference.sort_index()
        .equals(expected_a.sort_index())
    )
    paired = b.merge(meta, on="row_id", validate="one_to_one")
    assert paired.groupby("match_id").size().eq(2).all()
    for m in matches.itertuples():
        pair = paired[paired.match_id == m.match_id].set_index("team")
        assert pair.loc[m.team_a, "goals"] == m.goals_a
        assert pair.loc[m.team_b, "goals"] == m.goals_b
        assert np.allclose(
            pair[["elo_difference", "rest_difference", "home_advantage"]].sum(), 0
        )
    for row in pd.read_csv(ROOT / "audit/feature_audit.csv").itertuples():
        for field in ["team_a_history", "team_b_history"]:
            history = json.loads(getattr(row, field))
            assert len(history) == 10
            assert all(
                pd.Timestamp(h["date"]) < pd.Timestamp(row.date) for h in history
            )
            assert all(h["source_match"] != row.match_id for h in history)
    for model in ["2.1", "2.2"]:
        output = ROOT.parent / f"Regression_{model}/outputs"
        if not (output / "predictions.csv").exists():
            continue
        pred = pd.read_csv(output / "predictions.csv")
        target = "goal_difference" if model == "2.1" else "goals"
        metrics = pd.read_csv(output / "metrics.csv").set_index("method")
        for method in metrics.index:
            error = pred[target] - pred[method]
            assert np.isclose(abs(error).mean(), metrics.loc[method, "MAE"])
            assert np.isclose(np.sqrt((error**2).mean()), metrics.loc[method, "RMSE"])
        split = pd.read_csv(output / "split.csv")
        assert (
            split[split.split == "Training"].date.max()
            < split[split.split == "Test"].date.min()
        )
        cv = pd.read_csv(output / "cross_validation.csv")
        assert (cv.train_end < cv.validation_start).all()
        assert cv.validation_end.max() <= split[split.split == "Training"].date.max()
    print(
        "PASS: source hashes, 104 PDF headers, row counts, paired targets, prior-date histories and available model metrics."
    )


if __name__ == "__main__":
    validate()
