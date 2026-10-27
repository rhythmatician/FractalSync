"""Reject changed input while preserving the focused runner's source snapshot."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from unittest import mock


class Options(argparse.Namespace):
    source_root: Path
    target_dir: Path
    snapshot: Path
    output: Path


def stop_child(child: subprocess.Popen[str]) -> None:
    if child.poll() is None:
        if os.name == 'nt':
            # The PID belongs to this helper. Stop its compiler descendants too.
            try:
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                               capture_output=True, timeout=5, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
        else:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    try:
        child.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        if os.name == 'nt':
            try:
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                               capture_output=True, timeout=5, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
        else:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if child.poll() is None:
            child.kill()
        child.wait(timeout=2)
        child.communicate(timeout=2)


def check_timeout_cleanup() -> dict[str, bool | int | None]:
    code = ("import subprocess,sys,time; "
            "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            "print('spawned',flush=True); time.sleep(30)")
    child = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, start_new_session=os.name != 'nt')
    timed_out = False
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == 'spawned'
        try:
            child.communicate(timeout=0.01)
        except subprocess.TimeoutExpired:
            timed_out = True
    finally:
        stop_child(child)
    assert timed_out and child.poll() is not None
    return {'timed_out': timed_out, 'returncode': child.returncode,
            'descendant_output_closed': True}


def check_taskkill_retry() -> dict[str, bool | int | None]:
    original = subprocess.run
    attempts = 0

    def fail_once(command: list[str], *, capture_output: bool, timeout: int,
                  check: bool) -> subprocess.CompletedProcess[bytes]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError('injected taskkill failure')
        return original(command, capture_output=capture_output, timeout=timeout, check=check)

    with mock.patch.object(subprocess, 'run', side_effect=fail_once):
        result = check_timeout_cleanup()
    assert attempts == 2
    result['taskkill_attempts'] = attempts
    return result


parser = argparse.ArgumentParser()
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--target-dir', type=Path, required=True)
parser.add_argument('--snapshot', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args(namespace=Options())
cleanup_control = check_timeout_cleanup()
retry_control = check_taskkill_retry() if os.name == 'nt' else None
candidate = args.source_root.resolve()
snapshot = json.loads(args.snapshot.read_text())
assert snapshot['immutable_source_snapshot'] and not snapshot['source_drift']
args.output.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='geometry145-source-drift-', dir=args.output.parent.resolve()) as name:
    stage = Path(name)
    clone = stage / 'source'
    for relative, expected in snapshot['source_sha256'].items():
        original = candidate / relative
        raw = original.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, original
        path = clone / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    transient = stage / 'compiled'
    transient.mkdir()
    env = os.environ.copy()
    for key in ['TEMP', 'TMP', 'TMPDIR']:
        env[key] = str(transient)
    output = stage / 'result.json'
    checker = Path(__file__).with_name('geometry-145-generic-component-check.py').resolve()
    command = [sys.executable, str(checker), '--source-root', str(clone),
               '--target-dir', str(args.target_dir.resolve()), '--test-filter',
               'represented_shore_intervals', '--output', str(output)]
    child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=os.name != 'nt')
    try:
        deadline = time.monotonic() + 15
        while not list(transient.glob('geometry145-generic-*/main.rs')):
            assert child.poll() is None, 'checker terminated before snapshot control'
            assert time.monotonic() < deadline, 'checker did not publish its temporary harness'
            time.sleep(0.01)
        changed = clone / 'runtime-core/src/geometry_provider.rs'
        changed.write_bytes(changed.read_bytes() + b'\n// deliberate source-drift control\n')
        stdout, stderr = child.communicate(timeout=45)
    finally:
        stop_child(child)
    result = json.loads(output.read_text())
    assert child.returncode != 0 and result['stage'] == 'source-drift'
    assert result['immutable_source_snapshot'] is True and result['source_drift'] == [str(changed)]
    assert result['executed_tests'] == 1 and not result['mutations']
    record = {'returncode': child.returncode, 'snapshot_control': result, 'stdout': stdout,
              'stderr': stderr, 'cleanup_control': cleanup_control,
              'taskkill_retry_control': retry_control,
              'scope': 'Only a temporary source copy changed after harness emission.'}
    args.output.write_text(json.dumps(record, indent=2) + '\n')
print('source-drift control: immutable snapshot test passed; changed-input run refused')
