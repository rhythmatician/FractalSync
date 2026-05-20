from pathlib import Path


def test_frontend_shader_matches_shared():
    repo_root = Path(__file__).resolve().parents[2]
    shared = repo_root / "shared" / "shaders" / "julia.frag"
    frontend = repo_root / "frontend" / "src" / "lib" / "shaders" / "julia.frag"

    assert shared.exists(), f"Shared shader missing: {shared}"
    assert frontend.exists(), f"Frontend shader missing: {frontend}"

    s = shared.read_text(encoding="utf-8").strip()
    f = frontend.read_text(encoding="utf-8").strip()

    assert s == f, "Frontend shader must match shared shader (auto-sync failure)"
