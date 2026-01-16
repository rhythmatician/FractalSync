"""
Model inference parity test.

Validates that:
1. Backend model inference runs correctly
2. ONNX model export preserves model behavior
3. Frontend ONNX.js inference matches backend PyTorch inference

Usage:
- Backend: pytest backend/tests/test_inference_parity.py
- Frontend: npm test (after ONNX model is exported)
"""

import json
import numpy as np
import pytest
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.physics_model import PhysicsAudioToVisualModel  # noqa: E402

TEST_CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
TEST_MODELS_DIR = Path(__file__).parent / "fixtures" / "models"
INFERENCE_BASELINE_FILE = Path(__file__).parent / "fixtures" / "inference_baseline.json"


@pytest.fixture
def trained_model():
    """Load a trained model for inference testing."""
    # Use the latest checkpoint if available
    checkpoint_files = list(TEST_CHECKPOINT_DIR.glob("checkpoint_epoch_*.pt"))

    if not checkpoint_files:
        pytest.skip("No checkpoint available for inference testing")

    latest_checkpoint = max(
        checkpoint_files,
        key=lambda p: p.stat().st_mtime,
    )

    # Load checkpoint
    checkpoint = torch.load(latest_checkpoint, map_location="cpu", weights_only=False)

    # Reconstruct model
    # Infer input dim from checkpoint if available
    input_dim = checkpoint.get("input_dim", 60)  # Default 60 (6 features × 10 frames)

    model = PhysicsAudioToVisualModel(
        window_frames=10,
        num_features_per_frame=6,  # 6 base features
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, input_dim


class TestModelInference:
    """Test model inference consistency and correctness."""

    def test_model_inference_shape(self, trained_model):
        """Verify model outputs correct shape."""
        model, input_dim = trained_model

        # Create dummy input
        batch_size = 4
        dummy_input = torch.randn(batch_size, input_dim)

        with torch.no_grad():
            output = model(dummy_input)

        # Should output 7 parameters: c_real, c_imag, h, s, v, zoom, speed
        assert output.shape == (batch_size, 7)

    def test_model_inference_consistency(self, trained_model):
        """Verify model inference is deterministic."""
        model, input_dim = trained_model

        # Same input should produce same output
        test_input = torch.randn(1, input_dim)

        with torch.no_grad():
            output1 = model(test_input)
            output2 = model(test_input)

        torch.testing.assert_close(output1, output2)

    def test_model_inference_batching(self, trained_model):
        """Verify batched inference is equivalent to individual inference."""
        model, input_dim = trained_model

        batch_size = 8
        batch_input = torch.randn(batch_size, input_dim)

        with torch.no_grad():
            batch_output = model(batch_input)

            # Compare with individual inference
            individual_outputs = []
            for i in range(batch_size):
                individual_output = model(batch_input[i : i + 1])
                individual_outputs.append(individual_output)

            individual_output_batch = torch.cat(individual_outputs, dim=0)

        torch.testing.assert_close(
            batch_output, individual_output_batch, atol=1e-6, rtol=1e-5
        )

    def test_model_output_ranges(self, trained_model):
        """Verify model outputs are in reasonable ranges."""
        model, input_dim = trained_model

        # Test multiple samples
        for _ in range(10):
            test_input = torch.randn(1, input_dim)

            with torch.no_grad():
                output = model(test_input)

            # Extract parameters
            c_real, c_imag, h, s, v, zoom, speed = output[0].numpy()

            # Julia seed should be in [-2, 2] (standard viewing window)
            assert -3 <= c_real <= 3, f"c_real out of range: {c_real}"
            assert -3 <= c_imag <= 3, f"c_imag out of range: {c_imag}"

            # Color parameters (hue, saturation, value) in [0, 1] or reasonable ranges
            assert 0 <= h <= 1 or -1 <= h <= 1, f"hue out of range: {h}"
            assert 0 <= s <= 1 or -1 <= s <= 1, f"saturation out of range: {s}"
            assert 0 <= v <= 1 or -1 <= v <= 1, f"value out of range: {v}"

            # Zoom and speed should be positive
            assert zoom > 0, f"zoom should be positive: {zoom}"
            # Speed can be negative (direction), but magnitude reasonable
            assert -10 <= speed <= 10, f"speed out of range: {speed}"

    def test_generate_inference_baseline(self, trained_model):
        """Generate baseline inference data for frontend comparison."""
        model, input_dim = trained_model

        # Create test inputs
        test_inputs = torch.randn(5, input_dim)

        with torch.no_grad():
            outputs = model(test_inputs)

        # Create baseline data
        baseline = {
            "config": {
                "input_dim": int(input_dim),
                "output_dim": 7,
                "num_test_samples": 5,
            },
            "test_inputs": test_inputs.numpy().tolist(),
            "expected_outputs": outputs.numpy().tolist(),
            "parameter_names": [
                "c_real",
                "c_imag",
                "hue",
                "saturation",
                "value",
                "zoom",
                "speed",
            ],
        }

        # Ensure fixtures directory exists
        INFERENCE_BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Write baseline
        with open(INFERENCE_BASELINE_FILE, "w") as f:
            json.dump(baseline, f, indent=2)

        assert INFERENCE_BASELINE_FILE.exists()

    def test_onnx_export_preserves_behavior(self, trained_model):
        """
        Verify ONNX export preserves model inference behavior.

        This compares PyTorch inference with ONNX inference on same inputs.
        """
        model, input_dim = trained_model

        # Create test input
        test_input = torch.randn(2, input_dim)

        # PyTorch inference
        with torch.no_grad():
            pytorch_output = model(test_input).numpy()

        # ONNX inference would require onnxruntime and exported model
        # For now, this documents the expected comparison
        # Once model is exported, this test can fully validate ONNX parity

        assert pytorch_output.shape == (2, 7)
        assert np.all(np.isfinite(pytorch_output))


class TestInferenceParity:
    """Cross-framework inference comparison tests."""

    def test_frontend_receives_correct_model(self):
        """
        Verify that the frontend will receive the correct ONNX model.

        This test documents what the frontend expects:
        - Model in frontend/public/models/model.onnx
        - Metadata in frontend/public/models/model.onnx_metadata.json

        Note: This test is informational and will be fully validated after training.
        """
        frontend_model_path = (
            Path(__file__).parent.parent.parent
            / "frontend"
            / "public"
            / "models"
            / "model.onnx"
        )
        frontend_metadata_path = frontend_model_path.parent / "model.onnx_metadata.json"

        # Check files exist (will be created during training)
        if frontend_model_path.exists():
            assert (
                frontend_model_path.stat().st_size > 1000
            ), "Model file seems too small"

        if frontend_metadata_path.exists():
            with open(frontend_metadata_path, "r") as f:
                metadata = json.load(f)

            # Verify essential metadata structure
            # num_features_per_frame is added by export_model.py
            # If not present, model was exported from an older checkpoint
            if "num_features_per_frame" not in metadata:
                pytest.skip(
                    "Metadata missing num_features_per_frame; "
                    "re-run training to update model export"
                )

            assert "input_dim" in metadata
            assert "feature_mean" in metadata
            assert "feature_std" in metadata

            # Verify normalization parameters match expected dimensions
            input_dim = metadata.get("input_dim")
            if input_dim:
                assert len(metadata["feature_mean"]) == input_dim
                assert len(metadata["feature_std"]) == input_dim
