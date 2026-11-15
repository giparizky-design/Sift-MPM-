"""v4.16: dashboard server access control (audit v4.14 B01/B02/T04).

Runs the real DashboardHandler on an ephemeral loopback port, with ROOT_DIR pointed at a
temporary tree so no repository file is read or written.
"""
from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import scripts.dashboard_server as ds

TOKEN = "test-token-0123456789abcdef"
PAGE = b'<html><head><meta name="mpm-api-token" content="__MPM_API_TOKEN__" /></head><body>ok</body></html>'


@pytest.fixture()
def server(tmp_path, monkeypatch):
    (tmp_path / "ui").mkdir()
    (tmp_path / "ui" / "index.html").write_bytes(PAGE)
    (tmp_path / "outputs").mkdir()
    monkeypatch.setattr(ds, "ROOT_DIR", tmp_path)
    monkeypatch.setenv("MPM_DASHBOARD_TOKEN", TOKEN)
    monkeypatch.setattr(ds, "_API_TOKEN", None)
    # /api/runs/rename reads the catalog; keep it in memory.
    monkeypatch.setattr(ds, "load_runs_catalog", lambda: {"runs": []})
    monkeypatch.setattr(ds, "save_runs_catalog", lambda c: pytest.fail("catalog must not be written"))

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ds.DashboardHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield httpd.server_address[1], tmp_path
    httpd.shutdown()
    httpd.server_close()


def _request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    data = resp.read()
    hdrs = {k.lower(): v for k, v in resp.getheaders()}
    conn.close()
    return resp.status, hdrs, data


def _json_headers(token=TOKEN, ctype="application/json"):
    h = {"Content-Type": ctype}
    if token is not None:
        h["X-MPM-Token"] = token
    return h


def test_post_without_token_is_refused(server):
    port, _ = server
    status, hdrs, data = _request(port, "POST", "/api/train", body=b"{}", headers=_json_headers(token=None))
    assert status == 403
    assert hdrs.get("access-control-allow-origin") == "*"
    assert "X-MPM-Token" in json.loads(data)["error"]


def test_post_with_wrong_token_is_refused(server):
    port, _ = server
    status, _, _ = _request(port, "POST", "/api/train/stop", body=b"{}", headers=_json_headers(token="nope"))
    assert status == 403


def test_post_requires_json_content_type(server):
    port, _ = server
    status, _, _ = _request(port, "POST", "/api/rerank", body=b"{}", headers=_json_headers(ctype="text/plain"))
    assert status == 415


def test_post_rejects_bad_json_and_non_object(server):
    port, _ = server
    assert _request(port, "POST", "/api/rerank", body=b"{not json", headers=_json_headers())[0] == 400
    assert _request(port, "POST", "/api/rerank", body=b"[1, 2]", headers=_json_headers())[0] == 400


def test_authorized_post_reaches_route(server):
    port, _ = server
    status, _, data = _request(
        port, "POST", "/api/runs/rename", body=b"{}", headers=_json_headers(ctype="application/json; charset=utf-8")
    )
    assert status == 400
    assert "Missing run_id" in json.loads(data)["error"]


def test_options_preflight_allows_token_header(server):
    port, _ = server
    status, hdrs, _ = _request(port, "OPTIONS", "/api/train")
    assert status == 204
    assert "X-MPM-Token" in hdrs["access-control-allow-headers"]
    assert hdrs.get("access-control-allow-origin") == "*"


def test_index_injects_token_for_loopback_navigation(server):
    port, _ = server
    status, _, data = _request(port, "GET", "/")
    assert status == 200
    assert TOKEN.encode() in data
    assert b"__MPM_API_TOKEN__" not in data


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "http://evil.example"},
        {"Origin": "null"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Host": "evil.example:8765"},
    ],
)
def test_index_withholds_token_from_untrusted_requests(server, headers):
    port, _ = server
    status, hdrs, data = _request(port, "GET", "/", headers=headers)
    assert status == 200
    assert TOKEN.encode() not in data
    assert b"__MPM_API_TOKEN__" in data
    assert hdrs.get("access-control-allow-origin") == "*"


def test_head_content_length_matches_get(server):
    port, _ = server
    _, get_hdrs, get_body = _request(port, "GET", "/")
    _, head_hdrs, _ = _request(port, "HEAD", "/")
    assert int(head_hdrs["content-length"]) == len(get_body) == int(get_hdrs["content-length"])


def test_base_model_dir_restricted_to_outputs(server, tmp_path_factory):
    _, root = server
    outside = tmp_path_factory.mktemp("planted")
    (outside / "k4_ensemble.joblib").write_bytes(b"x")
    assert ds._resolve_base_model_dir(None, str(outside)) is None
    assert ds._resolve_base_model_dir(None, "../..") is None
    assert ds._resolve_base_model_dir(None, None, fallback_region="x/../../../../" + str(outside).lstrip("/")) is None

    inside = root / "outputs" / "seven_pass_east_kalimantan" / "models"
    inside.mkdir(parents=True)
    assert ds._resolve_base_model_dir(None, "outputs/seven_pass_east_kalimantan") == inside
    assert ds._resolve_base_model_dir(None, str(inside)) == inside


def test_base_model_dir_rejects_symlink_escape(server, tmp_path_factory):
    _, root = server
    outside = tmp_path_factory.mktemp("elsewhere")
    link = root / "outputs" / "sneaky"
    link.symlink_to(outside, target_is_directory=True)
    assert ds._resolve_base_model_dir(None, "outputs/sneaky") is None


def test_non_loopback_host_requires_allow_remote(monkeypatch):
    monkeypatch.setenv("MPM_DASHBOARD_TOKEN", TOKEN)
    with pytest.raises(SystemExit):
        ds.main(["--host", "0.0.0.0", "--port", "0"])


@pytest.mark.parametrize(
    ("host", "expected"),
    [("127.0.0.1:8765", True), ("localhost", True), ("[::1]:8765", True), ("evil.example", False), (None, False)],
)
def test_host_is_loopback(host, expected):
    assert ds._host_is_loopback(host) is expected


def test_token_file_created_with_0600(tmp_path, monkeypatch):
    monkeypatch.delenv("MPM_DASHBOARD_TOKEN", raising=False)
    monkeypatch.setattr(ds, "_API_TOKEN", None)
    tok_file = tmp_path / "cfg" / "dashboard_token"
    monkeypatch.setattr(ds, "API_TOKEN_FILE", tok_file)
    tok = ds._load_or_create_api_token()
    assert len(tok) >= 32
    assert tok_file.read_text().strip() == tok
    assert (tok_file.stat().st_mode & 0o777) == 0o600
    monkeypatch.setattr(ds, "_API_TOKEN", None)
    assert ds._load_or_create_api_token() == tok
