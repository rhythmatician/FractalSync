from pathlib import Path
p=Path('tmp_model.onnx')
if p.exists():
    b=p.open('rb').read(256)
    print('tmp exists', p.exists(), 'size', p.stat().st_size)
    print('hex16', b[:16].hex())
    print('preview', ''.join((chr(c) if 32 <= c < 127 else '.') for c in b[:120]))
else:
    print('tmp missing')
