from pathlib import Path
import onnx
files = sorted(Path('backend/checkpoints').glob('*.onnx'), key=lambda x: x.stat().st_mtime)
if not files:
    print('NO FILES')
else:
    p = files[-1]
    print('Trying', p)
    try:
        m = onnx.load(str(p))
        onnx.checker.check_model(m)
        print('ONNX model loads and checks OK')
    except Exception as e:
        print('ONNX failed:', type(e), e)
