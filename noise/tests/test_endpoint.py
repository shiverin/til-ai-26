"""POST /noise smoke test via FastAPI TestClient. Mocks NoiseManager
so the test doesn't require GPU."""
from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    class StubManager:
        def __init__(self):
            pass
        def noise(self, image: bytes) -> str:
            return base64.b64encode(image).decode("ascii")
    monkeypatch.setattr("src.noise_manager.NoiseManager", StubManager)
    import sys
    if "src.noise_server" in sys.modules:
        del sys.modules["src.noise_server"]
    from src import noise_server
    return TestClient(noise_server.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"message": "health ok"}


def test_noise_round_trip(client, synth_image_bytes):
    payload = {
        "instances": [{"b64": base64.b64encode(synth_image_bytes).decode("ascii")}]
    }
    r = client.post("/noise", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    raw = base64.b64decode(data["predictions"][0])
    assert raw == synth_image_bytes
