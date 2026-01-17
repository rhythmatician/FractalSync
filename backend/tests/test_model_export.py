"""Unit tests for model export to ONNX format."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.export_model import export_to_onnx, load_checkpoint_and_export  # noqa: E402
from src.physics_model import PhysicsAudioToVisualModel  # noqa: E402


class TestModelExport(unittest.TestCase):
    """Test ONNX export functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_export_to_onnx_creates_file(self):
        """Test that export_to_onnx creates an ONNX model file."""
        # Create a simple model
        model = PhysicsAudioToVisualModel(window_frames=10)
        model.eval()

        # Export to ONNX
        output_path = self.temp_path / "test_model.onnx"
        export_to_onnx(
            model,
            input_shape=(60,),
            output_path=str(output_path),
        )

        # Verify file exists
        self.assertTrue(
            output_path.exists(),
            f"ONNX file not created at {output_path}",
        )

        # Verify it's a valid ONNX model
        onnx_model = onnx.load(str(output_path))
        self.assertIsNotNone(onnx_model)

    def test_onnx_model_input_output_shape(self):
        """Test that exported ONNX model has correct input/output shape."""
        # Create and export model
        model = PhysicsAudioToVisualModel(window_frames=10)
        model.eval()

        output_path = self.temp_path / "test_io_shape.onnx"
        export_to_onnx(
            model,
            input_shape=(60,),
            output_path=str(output_path),
        )

        # Load ONNX model
        onnx_model = onnx.load(str(output_path))

        # Check input
        inputs = onnx_model.graph.input
        self.assertEqual(len(inputs), 1, "Should have exactly one input")
        self.assertEqual(inputs[0].name, "audio_features")

        # Check output
        outputs = onnx_model.graph.output
        self.assertEqual(len(outputs), 1, "Should have exactly one output")
        self.assertEqual(outputs[0].name, "visual_parameters")

    def test_export_metadata_saved(self):
        """Test that metadata is saved alongside ONNX model."""
        # Create and export model with metadata
        model = PhysicsAudioToVisualModel(window_frames=10)
        model.eval()

        output_path = self.temp_path / "test_metadata.onnx"
        metadata = {
            "epochs": 10,
            "window_frames": 10,
            "input_dim": 60,
        }

        export_to_onnx(
            model,
            input_shape=(60,),
            output_path=str(output_path),
            metadata=metadata,
        )

        # Verify metadata JSON exists
        metadata_path = output_path.with_suffix(".onnx_metadata.json")
        self.assertTrue(
            metadata_path.exists(),
            f"Metadata file not created at {metadata_path}",
        )

        # Verify metadata content
        with open(metadata_path) as f:
            saved_metadata = json.load(f)

        self.assertEqual(saved_metadata["epochs"], 10)
        self.assertEqual(saved_metadata["window_frames"], 10)
        self.assertEqual(saved_metadata["input_dim"], 60)

    def test_load_checkpoint_and_export(self):
        """Test loading checkpoint and exporting to ONNX."""
        # Create a model and checkpoint
        model = PhysicsAudioToVisualModel(window_frames=10)
        model.eval()

        checkpoint_path = self.temp_path / "test_checkpoint.pt"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": 5,
            "feature_mean": np.zeros(60),
            "feature_std": np.ones(60),
        }
        torch.save(checkpoint, checkpoint_path)

        # Load and export
        output_dir = self.temp_path / "export_output"
        output_dir.mkdir(exist_ok=True)

        onnx_path, metadata_path = load_checkpoint_and_export(
            str(checkpoint_path),
            output_dir=str(output_dir),
            window_frames=10,
        )

        # Verify files exist
        self.assertTrue(Path(onnx_path).exists(), "ONNX file not created")
        self.assertTrue(Path(metadata_path).exists(), "Metadata file not created")

        # Verify ONNX model is valid
        onnx_model = onnx.load(onnx_path)
        self.assertIsNotNone(onnx_model)

    def test_exported_model_output_shape(self):
        """Test that ONNX model produces correct output shape."""
        # Create model
        model = PhysicsAudioToVisualModel(window_frames=10)
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 60)

        # Get PyTorch output
        with torch.no_grad():
            torch_output = model(dummy_input).numpy()

        # Export to ONNX
        output_path = self.temp_path / "test_output_shape.onnx"
        export_to_onnx(
            model,
            input_shape=(60,),
            output_path=str(output_path),
        )

        # Verify output shape is (batch_size, 9) for PhysicsAudioToVisualModel
        # Output: [v_real, v_imag, c_real, c_imag, hue, sat, bright, zoom, speed]
        self.assertEqual(
            torch_output.shape,
            (1, 9),
            f"Expected output shape (1, 9), got {torch_output.shape}",
        )


if __name__ == "__main__":
    unittest.main()
