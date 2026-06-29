import unittest
import numpy as np
import torch
import os
from fastapi.testclient import TestClient

from src.preprocessing.preprocessor import EEGPreprocessor
from src.models.deep_learning import EEGNet, ShallowConvNet, DeepConvNet
from src.realtime.engine import StreamingBuffer, RealTimeInferenceEngine
from src.simulation.pathology import PathologicalSimulator
from src.transfer.adapter import ModelAdapter
from src.features.pipelines import CSP_LDA_Pipeline
from src.features.dl_pipeline import PyTorchPipeline
from src.api.server import app


class TestANDCPPipeline(unittest.TestCase):

    def setUp(self):
        self.n_channels = 22
        self.sfreq = 250.0
        self.n_times = 1000  # 4 seconds

    def test_preprocessor_normalization(self):
        """Unit test: Check EEGPreprocessor normalization."""
        preprocessor = EEGPreprocessor()
        dummy_epochs = np.random.randn(5, self.n_channels, self.n_times)
        norm_epochs = preprocessor.normalize_epochs(dummy_epochs)

        self.assertEqual(norm_epochs.shape, dummy_epochs.shape)
        # Means should be close to 0 along the time axis (last axis)
        np.testing.assert_array_almost_equal(
            np.mean(norm_epochs, axis=-1), np.zeros((5, self.n_channels)), decimal=4
        )
        # Stds should be close to 1
        np.testing.assert_array_almost_equal(
            np.std(norm_epochs, axis=-1), np.ones((5, self.n_channels)), decimal=4
        )

    def test_pytorch_models_output(self):
        """Unit test: Check Deep Learning networks compile and return (batch, n_classes) shape."""
        batch_size = 4
        dummy_input = torch.randn(batch_size, 1, self.n_channels, self.n_times)

        models = [
            EEGNet(n_channels=self.n_channels, n_classes=4, n_times=self.n_times),
            ShallowConvNet(
                n_channels=self.n_channels, n_classes=4, n_times=self.n_times
            ),
            DeepConvNet(n_channels=self.n_channels, n_classes=4, n_times=self.n_times),
        ]

        for model in models:
            output = model(dummy_input)
            self.assertEqual(output.shape, (batch_size, 4))

    def test_streaming_buffer_circularity(self):
        """Unit test: Verify that StreamingBuffer correctly wraps around circularly."""
        max_len = 100
        buffer = StreamingBuffer(n_channels=2, max_len=max_len, sfreq=self.sfreq)

        # Append chunk of 80 samples
        chunk1 = np.ones((80, 2)) * 5.0
        buffer.append(chunk1)

        # Append chunk of 40 samples (total 120, wraps around by 20)
        chunk2 = np.ones((40, 2)) * 10.0
        buffer.append(chunk2)

        self.assertEqual(buffer.total_written, 120)
        self.assertEqual(buffer.pointer, 20)

        # Retrieve latest 30 samples: should be all 10.0
        latest_30 = buffer.get_latest_window(30)
        np.testing.assert_array_equal(latest_30, np.ones((2, 30)) * 10.0)

        # Retrieve latest 50 samples: should be 40 samples of 10.0 (newest) and 10 samples of 5.0
        latest_50 = buffer.get_latest_window(50)
        expected = np.zeros((2, 50))
        expected[:, :10] = 5.0
        expected[:, 10:] = 10.0
        np.testing.assert_array_equal(latest_50, expected)

    def test_pathology_simulator(self):
        """Unit test: Verify that pathological simulator degrades signal without altering dimensions."""
        sim = PathologicalSimulator(
            fs=self.sfreq,
            emg_amplitude=2.0,
            drift_amplitude=1.5,
            electrode_shift_prob=0.2,
            gaussian_noise_std=0.5,
            dropout_prob=0.1,
        )
        dummy_signal = np.random.randn(200, self.n_channels)
        degraded = sim.simulate(dummy_signal)

        self.assertEqual(degraded.shape, dummy_signal.shape)
        # Check signal was modified
        self.assertFalse(np.array_equal(degraded, dummy_signal))

    def test_end_to_end_realtime_integration(self):
        """Integration test: Preprocessor + Model + Buffer + Pathology + Inference Engine."""
        # Create pipeline
        pipeline = CSP_LDA_Pipeline(n_components=4)

        # Train on mock data ensuring all 4 classes are present
        X_train = np.random.randn(12, self.n_channels, self.n_times)
        y_train = np.array([0, 1, 2, 3] * 3)
        pipeline.fit(X_train, y_train)

        # Buffer and Engine
        buffer = StreamingBuffer(
            n_channels=self.n_channels, max_len=1200, sfreq=self.sfreq
        )
        preprocessor = EEGPreprocessor()
        engine = RealTimeInferenceEngine(
            pipeline=pipeline,
            buffer=buffer,
            preprocessor=preprocessor,
            window_size_sec=4.0,
            sfreq=self.sfreq,
        )

        # Stream 1100 samples
        chunk = np.random.randn(1100, self.n_channels)
        buffer.append(chunk)

        # Run inference step
        pred, probs = engine.run_inference_step()
        self.assertIn(pred, [0, 1, 2, 3])
        self.assertEqual(len(probs), 4)
        np.testing.assert_almost_equal(np.sum(probs), 1.0, decimal=5)

    def test_model_serialization_exports(self):
        """Unit test: Verify ONNX and TorchScript serialization exports for PyTorch pipelines."""
        model_kwargs = {
            "n_channels": self.n_channels,
            "n_classes": 4,
            "n_times": self.n_times,
        }
        pipeline = PyTorchPipeline(
            model_class=EEGNet,
            model_kwargs=model_kwargs,
            epochs=1,
            batch_size=2,
            lr=1e-3,
        )

        # Fit on mock data
        X = np.random.randn(2, self.n_channels, self.n_times)
        y = np.array([0, 1])
        pipeline.fit(X, y)

        onnx_temp = "tests/temp_model.onnx"
        ts_temp = "tests/temp_model_ts.pt"

        try:
            pipeline.export_onnx(onnx_temp)
            self.assertTrue(os.path.exists(onnx_temp))

            pipeline.export_torchscript(ts_temp)
            self.assertTrue(os.path.exists(ts_temp))
        finally:
            if os.path.exists(onnx_temp):
                os.remove(onnx_temp)
            if os.path.exists(ts_temp):
                os.remove(ts_temp)

    def test_api_rest_endpoints(self):
        """Unit test: Verify Health, Config, Metrics, and Predict FastAPI endpoints."""
        # Use context manager to trigger FastAPI startup event
        with TestClient(app) as client:
            # 1. Test GET /health
            resp_health = client.get("/health")
            self.assertEqual(resp_health.status_code, 200)
            self.assertEqual(resp_health.json()["status"], "healthy")

            # 2. Test GET /config
            resp_config = client.get("/config")
            self.assertEqual(resp_config.status_code, 200)
            self.assertIn("preprocessing", resp_config.json())

            # 3. Test GET /metrics
            resp_metrics = client.get("/metrics")
            self.assertEqual(resp_metrics.status_code, 200)
            self.assertEqual(resp_metrics.json()["status"], "active")
            self.assertIn("cpu_usage_percent", resp_metrics.json())

            # 4. Test POST /predict
            dummy_epoch = np.random.randn(22, 1000).tolist()
            resp_predict = client.post("/predict", json={"data": dummy_epoch})
            self.assertEqual(resp_predict.status_code, 200)
            self.assertIn("prediction_class", resp_predict.json())
            self.assertEqual(len(resp_predict.json()["probabilities"]), 4)

    def test_calibration_workflow(self):
        """Unit test: Verify ModelAdapter linear_probe, fine_tune, and weight_interpolation."""
        pipeline = CSP_LDA_Pipeline(n_components=2)
        X_train = np.random.randn(8, self.n_channels, self.n_times)
        y_train = np.array([0, 1] * 4)
        pipeline.fit(X_train, y_train)

        adapter = ModelAdapter()
        X_cal = np.random.randn(4, self.n_channels, self.n_times)
        y_cal = np.array([0, 1, 0, 1])

        # Test linear probing
        adapted_lp = adapter.linear_probe(pipeline, X_cal, y_cal)
        self.assertIsNotNone(adapted_lp)

        # Test fine-tuning
        adapted_ft = adapter.fine_tune(pipeline, X_cal, y_cal)
        self.assertIsNotNone(adapted_ft)

        # Test weight interpolation
        target_fit = CSP_LDA_Pipeline(n_components=2)
        target_fit.fit(X_cal, y_cal)
        adapted_wi = adapter.interpolate_weights(pipeline, target_fit, alpha=0.5)
        self.assertIsNotNone(adapted_wi)

    def test_full_system_integration_flow(self):
        """Full System Test: Mock Raw Data -> Preprocessing -> Model -> API REST & WebSocket pipeline."""
        with TestClient(app) as client:
            mock_raw_data = np.random.randn(1100, self.n_channels)

            # Test live streaming WebSocket loop
            with client.websocket_connect("/stream") as websocket:
                # Send first chunk (50 samples)
                chunk1 = mock_raw_data[:50].tolist()
                websocket.send_json(chunk1)
                resp1 = websocket.receive_json()
                self.assertEqual(resp1["status"], "buffering")

                # Stream the rest of the trial to trigger inference
                for start in range(50, 1100, 50):
                    end = start + 50
                    websocket.send_json(mock_raw_data[start:end].tolist())
                    resp = websocket.receive_json()
                    if "prediction_label" in resp:
                        self.assertIn("prediction_class", resp)
                        self.assertEqual(len(resp["probabilities"]), 4)
                        break


if __name__ == "__main__":
    unittest.main()
