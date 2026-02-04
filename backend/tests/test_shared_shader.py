import sys
from pathlib import Path

# Ensure backend/src is on sys.path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from src.shaders import read_shader  # noqa: E402


def test_julia_shader_contains_required_symbols():
    txt = read_shader("julia.frag")
    # Ensure we have functions that our renderer and tests expect
    assert "potentialAt" in txt, "potentialAt missing"
    assert "potentialGradient" in txt, "potentialGradient missing"
    assert "smoothIterJulia" in txt, "smoothIterJulia missing"
    assert "shadePixel" in txt, "shadePixel missing"


def test_shader_endpoint_returns_shader():
    from fastapi.testclient import TestClient
    from api.server import app

    client = TestClient(app)
    resp = client.get("/api/shader/julia.frag")
    assert resp.status_code == 200
    assert "smoothIterJulia" in resp.text
