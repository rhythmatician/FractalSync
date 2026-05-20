import onnxruntime as ort
from pathlib import Path
p=sorted(Path('backend/checkpoints').glob('*.onnx'), key=lambda x:x.stat().st_mtime)[-1]
print('Trying',p)
try:
    sess=ort.InferenceSession(str(p))
    print('onnxruntime native session created; inputs:', [i.name for i in sess.get_inputs()])
except Exception as e:
    print('onnxruntime failed:', type(e), e)
