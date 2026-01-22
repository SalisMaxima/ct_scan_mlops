"""Tests for Streamlit frontend helpers."""

from __future__ import annotations

from dataclasses import dataclass

import ct_scan_mlops.frontend.pages.home as home


@dataclass
class DummyUpload:
    name: str = "scan.png"
    type: str = "image/png"

    def getvalue(self) -> bytes:
        return b"fake-image-bytes"


class DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class DummyClient:
    def __init__(self, capture: dict):
        self.capture = capture

    def __enter__(self) -> DummyClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, files: dict, data: dict):
        self.capture["url"] = url
        self.capture["files"] = files
        self.capture["data"] = data
        return DummyResponse({"ok": True})


def test_send_feedback_correct(monkeypatch):
    capture: dict = {}

    def _client_factory(timeout: int):
        return DummyClient(capture)

    monkeypatch.setattr(home.httpx, "Client", _client_factory)

    result = home._send_feedback(
        api_base_url="http://localhost:8000",
        uploaded_file=DummyUpload(),
        predicted_class="normal",
        is_correct=True,
        correct_class=None,
    )

    assert result == {"ok": True}
    assert capture["url"].endswith("/feedback")
    assert capture["data"]["predicted_class"] == "normal"
    assert capture["data"]["is_correct"] == "true"
    assert "correct_class" not in capture["data"]


def test_send_feedback_incorrect(monkeypatch):
    capture: dict = {}

    def _client_factory(timeout: int):
        return DummyClient(capture)

    monkeypatch.setattr(home.httpx, "Client", _client_factory)

    result = home._send_feedback(
        api_base_url="http://localhost:8000",
        uploaded_file=DummyUpload(),
        predicted_class="normal",
        is_correct=False,
        correct_class="adenocarcinoma",
    )

    assert result == {"ok": True}
    assert capture["data"]["is_correct"] == "false"
    assert capture["data"]["correct_class"] == "adenocarcinoma"
