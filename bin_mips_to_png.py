import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_mips(bin_path: Path, meta: dict, key: str) -> list[np.ndarray]:
    """
    key is "F" or "S" matching meta["F"] / meta["S"].
    Returns list of float32 arrays, one per mip, shape (H, W).
    """
    info = meta[key]
    widths = info["mip_widths"]
    heights = info["mip_heights"]
    offsets = info["mip_offsets_bytes"]
    levels = info["mip_levels"]

    data = bin_path.read_bytes()
    mips = []

    for lvl in range(levels):
        w = int(widths[lvl])
        h = int(heights[lvl])
        off = int(offsets[lvl])
        nbytes = w * h * 4  # f32

        chunk = data[off : off + nbytes]
        if len(chunk) != nbytes:
            raise ValueError(
                f"{key} mip {lvl}: expected {nbytes} bytes at offset {off}, got {len(chunk)}"
            )

        arr = np.frombuffer(chunk, dtype="<f4").reshape((h, w))
        mips.append(arr.copy())  # detach from backing buffer for safety

    return mips


def to_u8_grayscale(
    arr: np.ndarray, vmin: float = 0.0, vmax: float = 1.0
) -> np.ndarray:
    """
    Map float array to uint8 [0,255] using clamp to [vmin, vmax].
    """
    x = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def save_mips_png(mips: list[np.ndarray], out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, arr in enumerate(mips):
        img_u8 = to_u8_grayscale(arr, 0.0, 1.0)
        im = Image.fromarray(img_u8, mode="L")
        out_path = out_dir / f"{prefix}_mip{i:02d}_{arr.shape[1]}x{arr.shape[0]}.png"
        im.save(out_path)


def main():
    meta_path = Path("mandel_mips_meta.json")
    if not meta_path.exists():
        raise FileNotFoundError("mandel_mips_meta.json not found in current directory")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    out_root = Path("out_png")

    # F
    f_bin = Path("mandel_F_mips_f32.bin")
    if f_bin.exists():
        F_mips = read_mips(f_bin, meta, "F")
        save_mips_png(F_mips, out_root / "F", "F")
        print(f"Wrote {len(F_mips)} F mip PNGs to {out_root / 'F'}")
    else:
        print("mandel_F_mips_f32.bin not found; skipping F")

    # S
    s_bin = Path("mandel_S_mips_f32.bin")
    if s_bin.exists():
        S_mips = read_mips(s_bin, meta, "S")
        save_mips_png(S_mips, out_root / "S", "S")
        print(f"Wrote {len(S_mips)} S mip PNGs to {out_root / 'S'}")
    else:
        print("mandel_S_mips_f32.bin not found; skipping S")


if __name__ == "__main__":
    main()
