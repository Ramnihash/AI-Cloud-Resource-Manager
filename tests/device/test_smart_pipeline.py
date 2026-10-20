"""Tests for the Backblaze SMART pipeline: dataset builder, classifier service, study and API."""

import json
import tempfile
import unittest
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data.device.smart.build_failure_windows import FEATURE_COLUMNS, build_failure_windows, meta_path_for, save
from ml.device.smart import service

# ---- synthetic Backblaze-format fixture (not real data) ----


def write_fake_quarter(
    folder: Path,
    days: int = 40,
    drives: int = 120,
    failing: int = 12,
    as_zip: bool = False,
    seed: int = 0,
) -> Path:
    """Write ``days`` daily CSVs; ``failing`` drives fail on staggered days with rising SMART 5/187/197."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    start = date(2025, 1, 1)
    fail_day = {f"F{i:03d}": 15 + (i * 2) % (days - 15) for i in range(failing)}
    serials = [f"H{i:04d}" for i in range(drives - failing)] + list(fail_day)
    paths = []
    for d in range(days):
        rows = []
        for serial in serials:
            fd = fail_day.get(serial)
            if fd is not None and d > fd:
                continue  # a failed drive leaves the fleet
            wear = 0 if fd is None else max(0, 30 - (fd - d))
            rows.append({
                "date": (start + timedelta(days=d)).isoformat(),
                "serial_number": serial,
                "model": "ST4000DM000" if serial.endswith(("0", "2", "4")) else "HGST HMS5C4040BLE640",
                "capacity_bytes": 4000787030016,
                "failure": int(fd == d),
                "smart_5_raw": wear * 3 + (rng.integers(0, 2) if fd is None else 0),
                "smart_9_raw": 10000 + d * 24 + rng.integers(0, 500),
                "smart_187_raw": wear if serial.startswith("F") else np.nan,
                "smart_194_raw": 25 + rng.integers(0, 10),
                "smart_197_raw": wear // 2,
                "smart_241_raw": "" if d % 7 == 0 else rng.integers(10**9, 10**10),
            })
        path = folder / f"{(start + timedelta(days=d)).isoformat()}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        paths.append(path)
    if not as_zip:
        return folder
    archive = folder.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w") as zf:
        for path in paths:
            zf.write(path, arcname=f"data_Q1_2025/{path.name}")
    return archive


class TestBuildFailureWindows(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_labels_horizon_and_samples_healthy(self):
        source = write_fake_quarter(self.root / "q", days=40, drives=60, failing=6)
        data, stats = build_failure_windows(source, horizon_days=10, negatives_per_day=5)
        positives = data[data["failure"] == 1]
        self.assertTrue(positives["serial_number"].str.startswith("F").all())
        self.assertTrue(positives["days_to_failure"].between(0, 9).all())
        negatives = data[data["failure"] == 0]
        self.assertTrue(negatives["serial_number"].str.startswith("H").all())
        self.assertEqual(len(negatives), 40 * 5)
        self.assertEqual(stats["failed_drives"], 6)
        self.assertEqual(stats["files"], 40)
        self.assertGreater(stats["prevalence"], 0)
        self.assertEqual(list(data.columns[3:-1]), ["days_to_failure", *FEATURE_COLUMNS])
        # attributes missing from the raw files become NaN columns, blanks become NaN
        self.assertTrue(data["smart_1_raw"].isna().all())
        self.assertTrue(data["smart_241_raw"].isna().any())

    def test_zip_source_matches_folder(self):
        folder = write_fake_quarter(self.root / "q", days=25, drives=40, failing=4)
        archive = write_fake_quarter(self.root / "z", days=25, drives=40, failing=4, as_zip=True)
        a, _ = build_failure_windows(folder, negatives_per_day=3)
        b, _ = build_failure_windows(archive, negatives_per_day=3)
        pd.testing.assert_frame_equal(a, b)

    def test_rejects_bad_source(self):
        with self.assertRaises(ValueError):
            build_failure_windows(self.root / "missing")
        (self.root / "x.txt").write_text("hi")
        with self.assertRaises(ValueError):
            build_failure_windows(self.root / "x.txt")

    def test_save_writes_meta(self):
        source = write_fake_quarter(self.root / "q", days=20, drives=30, failing=3)
        data, stats = build_failure_windows(source, negatives_per_day=3)
        out, meta = save(data, stats, self.root / "out" / "set.csv.gz")
        self.assertEqual(meta, meta_path_for(out))
        self.assertEqual(json.loads(meta.read_text())["rows"], len(data))
        self.assertEqual(len(pd.read_csv(out)), len(data))


class TestSmartService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        source = write_fake_quarter(root / "q", days=45, drives=150, failing=20)
        data, stats = build_failure_windows(source, negatives_per_day=20)
        cls.path, _ = save(data, stats, root / "set.csv.gz")
        cls.path = str(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_predicts_higher_risk_for_worn_drive(self):
        healthy = service.predict_risk({"smart_5_raw": 0, "smart_197_raw": 0}, path=self.path)
        worn = service.predict_risk({"smart_5_raw": 80, "smart_187_raw": 25, "smart_197_raw": 12}, path=self.path)
        self.assertLess(healthy["failure_probability"], worn["failure_probability"])
        self.assertTrue(worn["at_risk"])
        self.assertIn("smart_9_raw", worn["attributes_imputed"])
        self.assertEqual(worn["horizon_days"], 30)

    def test_missing_dataset_raises(self):
        with self.assertRaises(service.SmartModelUnavailableError):
            service.get_model(path=str(Path(self.tmp.name) / "nope.csv.gz"))

    def test_validate_attributes(self):
        self.assertIsNotNone(service.validate_attributes({})[1])
        self.assertIsNotNone(service.validate_attributes({"smart_999_raw": 1})[1])
        self.assertIsNotNone(service.validate_attributes({"smart_5_raw": "x"})[1])
        self.assertIsNotNone(service.validate_attributes({"smart_5_raw": -1})[1])
        self.assertIsNotNone(service.validate_attributes({"smart_5_raw": float("nan")})[1])
        self.assertIsNotNone(service.validate_attributes({"smart_5_raw": None})[1])
        clean, err = service.validate_attributes({"smart_5_raw": 3, "smart_9_raw": None})
        self.assertIsNone(err)
        self.assertEqual(clean, {"smart_5_raw": 3.0})

    def test_study_runs_and_writes_results(self):
        from experiments import smart_eval

        with tempfile.TemporaryDirectory() as out:
            r = smart_eval.run(dataset_path=self.path, out=out, n_folds=3)
            self.assertEqual(set(r["summary"]["model"]), set(smart_eval.MODELS))
            self.assertEqual(len(r["folds"]), 3 * len(smart_eval.MODELS))
            for name in ("smart_folds.csv", "smart_metrics.csv", "smart_recall_by_lead.csv",
                         "smart_feature_importance.csv", "smart_roc_pr.png"):
                self.assertTrue((Path(out) / name).exists(), name)
            forest = r["summary"].set_index("model").loc["random_forest"]
            self.assertGreater(forest["roc_auc_mean"], 0.8)
            # folds are split by drive: no serial in two test folds
            self.assertTrue(r["folds"]["test_failed_drives"].gt(0).all())


class TestSmartApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from backend.app import create_app

        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        source = write_fake_quarter(root / "q", days=40, drives=100, failing=12)
        data, stats = build_failure_windows(source, negatives_per_day=15)
        path, _ = save(data, stats, root / "set.csv.gz")
        cls.original = service.DATASET_PATH
        service.DATASET_PATH = str(path)
        cls.client = create_app(test_config={"TESTING": True}).test_client()

    @classmethod
    def tearDownClass(cls):
        service.DATASET_PATH = cls.original
        cls.tmp.cleanup()

    def test_smart_risk_endpoint(self):
        r = self.client.post("/api/v1/device/smart-risk", json={"smart_attributes": {"smart_5_raw": 90, "smart_197_raw": 14}})
        self.assertEqual(r.status_code, 200, r.get_json())
        body = r.get_json()
        self.assertIn("failure_probability", body)
        self.assertEqual(body["model"], "random_forest")

    def test_smart_risk_rejects_bad_input(self):
        self.assertEqual(self.client.post("/api/v1/device/smart-risk", json={"x": 1}).status_code, 400)
        r = self.client.post("/api/v1/device/smart-risk", json={"smart_attributes": {"smart_5_raw": "a"}})
        self.assertEqual(r.status_code, 400)

    def test_predictive_action_uses_smart_model(self):
        r = self.client.post("/api/v1/device/predictive-action", json={
            "device_id": "disk-1", "device_status": "degrading",
            "smart_attributes": {"smart_5_raw": 90, "smart_187_raw": 28, "smart_197_raw": 14},
        })
        self.assertEqual(r.status_code, 200, r.get_json())
        body = r.get_json()
        self.assertAlmostEqual(body["failure_probability"], body["smart_prediction"]["failure_probability"])
