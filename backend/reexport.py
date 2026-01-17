from src.export_model import load_checkpoint_and_export
import os
import sys
import argparse

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Re-export model to ONNX with opset 11")
parser.add_argument("--cp", type=str, help="Path to the model checkpoint file")
args = parser.parse_args()
checkpoint_path = args.cp

load_checkpoint_and_export(checkpoint_path, window_frames=10)
print("✓ Model re-exported with opset 11")
