import subprocess
import pytest
import re
import os
from argparse import Namespace
from pathlib import Path
from train import execute_training_workflow
from scripts.remove_epoch1_models import main as clean_up


def test_trainer_e2e():
    command = "python train.py  --data-dir data/testing"
    # Use the backend directory (calculated from this test file) so the test
    # behaves the same regardless of where pytest is invoked from.
    working_dir = str(Path(__file__).resolve().parent.parent)

    result = subprocess.run(
        command,
        shell=True,
        cwd=working_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )

    return_code = result.returncode
    assert return_code == 0, f"Trainer script failed with return code {return_code}"

    result_stdout = result.stdout.splitlines()
    errors = []
    for line in result_stdout:
        if line.startswith("[ERROR]"):
            errors.append(line)
    assert not errors, f"Errors were found in trainer script output: {errors}"

    model_exported = False
    for line in result_stdout:
        print(line)
        if not line.startswith("[INFO] Model exported successfully to "):
            continue

        # Match either the old logging format or the current print
        match = re.search(
            r"\[INFO\] Model exported successfully to (.+?) \(opset=\d+\)", line
        ) or re.search(r"Model exported to: (.+)", line)
        if not match:
            continue
        path = match.group(1).strip()
        # Make path absolute relative to the backend working dir so checks
        # work whether pytest was invoked from repo root or the backend dir.
        if not os.path.isabs(path):
            path = os.path.join(working_dir, path)
        try:
            with open(path, "rb"):
                pass
            model_exported = True
        except FileNotFoundError:
            pytest.fail(f"Exported model file not found at {path}")
        break
    assert model_exported, "Model export message not found in output"

    assert any(
        line.startswith("[OK] Training complete!") for line in result_stdout
    ), "Trainer script did not complete successfully"

    # Clean up epoch 1 models created during the test
    clean_up()


def test_execute_training_workflow(monkeypatch, tmp_path, capsys):
    """Call the public workflow function but stub heavy components so the
    test runs quickly and deterministically.
    """

    backend_dir = Path(__file__).resolve().parent.parent

    args = Namespace(
        data_dir=str(backend_dir / "data" / "testing"),
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        window_frames=1,
        k_bands=1,
        use_curriculum=False,
        curriculum_weight=1.0,
        curriculum_decay=0.5,
        device="cpu",
        no_gpu_rendering=True,
        julia_resolution=16,
        julia_max_iter=10,
        num_workers=0,
        max_files=1,
        save_dir=str(tmp_path / "checkpoints"),
    )
    try:
        execute_training_workflow(args)
    except Exception as e:
        pytest.fail(f"execute_training_workflow raised an exception: {e}")

    finally:
        # Cleanup any created files
        if Path(args.save_dir).exists():
            for f in Path(args.save_dir).iterdir():
                f.unlink()
            Path(args.save_dir).rmdir()
