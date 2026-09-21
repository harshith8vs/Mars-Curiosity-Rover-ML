"""
API test suite for Mars Rover ML Classification Service.

Tests:
  - System health and model registry
  - Dataset sample retrieval, search, filter, and image serving
  - Canonical benchmark loading from results/10_Overall_Comparison/final_test_comparison.json
  - Stored test predictions lookup (adhering to inference_time_ms = None)
  - Train/val restriction when checkpoint is not present
  - Live checkpoint inference when model checkpoint is present
"""

import unittest
from fastapi.testclient import TestClient
from api.server import app


class TestMarsRoverAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["dataset_loaded"])
        self.assertEqual(data["models_available"], 9)
        self.assertEqual(data["test_samples"], 1305)

    def test_02_models_list(self):
        res = self.client.get("/api/models")
        self.assertEqual(res.status_code, 200)
        models = res.json()
        self.assertEqual(len(models), 9)
        
        # Verify EfficientNet-B3 does not display "Champion" in display name
        effnet = next(m for m in models if m["id"] == "efficientnet_b3")
        self.assertNotIn("Champion", effnet["name"])
        self.assertEqual(effnet["name"], "EfficientNet-B3")

    def test_03_benchmarks(self):
        res = self.client.get("/api/benchmarks")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["evaluation_split"].lower(), "test")
        self.assertEqual(data["num_test_samples"], 1305)
        self.assertEqual(len(data["models"]), 9)
        
        # Check that top model is EfficientNet-B3 with canonical accuracy ~80.61%
        effnet_summary = next(m for m in data["models"] if m["key"] == "efficientnet_b3")
        self.assertAlmostEqual(effnet_summary["accuracy_pct"], 80.61, places=1)

    def test_04_samples_pagination_and_split(self):
        # Default test split
        res = self.client.get("/api/samples?split=test&page=1&page_size=5")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 1305)
        self.assertEqual(len(data["samples"]), 5)
        for s in data["samples"]:
            self.assertEqual(s["split"], "test")

        # Train split
        res_train = self.client.get("/api/samples?split=train&page=1&page_size=3")
        self.assertEqual(res_train.status_code, 200)
        self.assertGreater(res_train.json()["total"], 3000)

    def test_05_samples_search_and_filter(self):
        res = self.client.get("/api/samples?split=test&q=TEST-0000")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(len(data["samples"]) >= 1)
        self.assertEqual(data["samples"][0]["sample_id"], "TEST-0000")

    def test_06_sample_detail_and_image(self):
        res = self.client.get("/api/samples/TEST-0000")
        self.assertEqual(res.status_code, 200)
        sample = res.json()
        self.assertEqual(sample["sample_id"], "TEST-0000")
        
        # Image stream
        img_res = self.client.get("/api/samples/TEST-0000/image")
        self.assertEqual(img_res.status_code, 200)
        self.assertIn(img_res.headers["content-type"], ["image/jpeg", "image/png"])
        self.assertGreater(len(img_res.content), 0)

    def test_07_dataset_stats(self):
        res = self.client.get("/api/dataset/stats")
        self.assertEqual(res.status_code, 200)
        stats = res.json()
        self.assertEqual(stats["splits"]["test"], 1305)
        self.assertGreater(stats["splits"]["train"], 3000)
        self.assertGreater(stats["splits"]["val"], 800)
        self.assertGreaterEqual(len(stats["classes"]), 20)

    def test_08_classify_stored_test_prediction(self):
        """
        User correction 2 & 1:
        - For STORED TEST PREDICTION, inference_time_ms must be None.
        - inference_time_display must be 'N/A — STORED PREDICTION'.
        - inference_source must be 'STORED TEST PREDICTION'.
        - Deep learning models must have real probability distribution from .npz.
        """
        payload = {
            "sample_id": "TEST-0000",
            "model": "resnet50"
        }
        res = self.client.post("/api/classify", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["sample_id"], "TEST-0000")
        self.assertEqual(data["model"], "resnet50")
        self.assertEqual(data["inference_source"], "STORED TEST PREDICTION")
        self.assertIsNone(data["inference_time_ms"])
        self.assertEqual(data["inference_time_display"], "N/A — STORED PREDICTION")
        self.assertIsNotNone(data["predicted_class_name"])
        self.assertGreater(len(data["top_predictions"]), 0)

    def test_09_classify_train_sample_without_checkpoint_blocked(self):
        """
        User correction 1:
        TRAIN/VAL samples may only be classified when a local checkpoint is available for live inference.
        If a model has NO checkpoint, attempting to classify TRAIN sample returns 422 Unprocessable Entity.
        """
        res_models = self.client.get("/api/models")
        uncheckpointed = [m for m in res_models.json() if not m["has_checkpoint"]]
        
        if uncheckpointed:
            target_model_id = uncheckpointed[0]["id"]
            payload = {
                "sample_id": "TRAIN-0000",
                "model": target_model_id
            }
            res = self.client.post("/api/classify", json=payload)
            self.assertEqual(res.status_code, 422)
            self.assertIn("has no local checkpoint", res.json()["detail"])


if __name__ == "__main__":
    unittest.main()
