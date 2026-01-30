import subprocess
import pytest
import re


def test_trainer_e2e():
    command = "python train.py  --data-dir data/testing"
    working_dir = "backend"

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
    assert not any(
        line.startswith("[Error]") for line in result_stdout
    ), "Trainer script output contains errors"

    model_exported = False
    for line in result_stdout:
        if not line.startswith("[INFO] Model exported successfully to "):
            continue

        match = re.search(
            r"\[INFO\] Model exported successfully to (.+?) \(opset=\d+\)", line
        )
        if not match:
            continue
        path = "backend/" + match.group(1)
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
