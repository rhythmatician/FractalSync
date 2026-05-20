from pathlib import Path
files = sorted(Path('backend/checkpoints').glob('*.onnx'), key=lambda x: x.stat().st_mtime)
if not files:
    print('NO FILES')
else:
    f = files[-1]
    print('file', f)
    b = f.open('rb').read(512)
    print('len', len(b))
    print('hex16', b[:16].hex())
    print('start8', b[:8])
    # Print first 80 bytes as ASCII-safe
    print('preview', ''.join((chr(c) if 32 <= c < 127 else '.') for c in b[:80]))
