import subprocess


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

    assert any(
        line.startswith("[INFO] Model exported successfully to ")
        for line in result_stdout
    ), "Trainer script did not complete successfully"

    assert any(
        line.startswith("[OK] Training complete!") for line in result_stdout
    ), "Trainer script did not complete successfully"
