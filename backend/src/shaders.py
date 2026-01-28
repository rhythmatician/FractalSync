from pathlib import Path

SHADERS_DIR = Path(__file__).resolve().parents[1].parent / "shared" / "shaders"


def get_shader_path(name: str) -> Path:
    # simple whitelist to avoid path traversal
    allowed = {"julia.frag"}
    if name not in allowed:
        raise FileNotFoundError(f"Shader '{name}' is not available")
    path = SHADERS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Shader file not found: {path}")
    return path


def read_shader(name: str) -> str:
    path = get_shader_path(name)
    return path.read_text(encoding="utf-8")
