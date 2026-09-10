import threading
import pytest
import threading
import time
from utils import *

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.router()


def test_router_props():
    global server
    server.models_max = 2
    server.no_models_autoload = True
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["role"] == "router"
    assert res.body["max_instances"] == 2
    assert res.body["models_autoload"] is False
    assert res.body["build_info"].startswith("b")


@pytest.mark.parametrize(
    "model,success",
    [
        ("ggml-org/tinygemma3-GGUF:Q8_0", True),
        ("non-existent/model", False),
    ]
)
def test_router_chat_completion_stream(model: str, success: bool):
    global server
    server.start()
    content = ""
    ex: ServerError | None = None
    try:
        res = server.make_stream_request("POST", "/chat/completions", data={
            "model": model,
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "hello"},
            ],
            "stream": True,
        })
        for data in res:
            if data["choices"]:
                choice = data["choices"][0]
                if choice["finish_reason"] in ["stop", "length"]:
                    assert "content" not in choice["delta"]
                else:
                    assert choice["finish_reason"] is None
                    content += choice["delta"]["content"] or ''
    except ServerError as e:
        ex = e

    if success:
        assert ex is None
        assert len(content) > 0
    else:
        assert ex is not None
        assert content == ""


def _get_model_ids(is_reload: bool, headers: dict | None = None) -> set[str]:
    res = server.make_request(
        "GET", "/models" + ("?reload=1" if is_reload else ""), headers=headers
    )
    assert res.status_code == 200
    return {item["id"] for item in res.body.get("data", [])}


def _get_model_status(model_id: str, headers: dict | None = None) -> str:
    res = server.make_request("GET", "/models", headers=headers)
    assert res.status_code == 200
    for item in res.body.get("data", []):
        if item.get("id") == model_id or item.get("model") == model_id:
            return item["status"]["value"]
    raise AssertionError(f"Model {model_id} not found in /models response")


def _wait_for_model_status(model_id: str, desired: set[str], timeout: int = 60, headers: dict | None = None) -> str:
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        last_status = _get_model_status(model_id, headers=headers)
        if last_status in desired:
            return last_status
        time.sleep(0.01)
    raise AssertionError(
        f"Timed out waiting for {model_id} to reach {desired}, last status: {last_status}"
    )


def _load_model_and_wait(
    model_id: str, timeout: int = 60, headers: dict | None = None
) -> None:
    load_res = server.make_request(
        "POST", "/models/load", data={"model": model_id}, headers=headers
    )
    assert load_res.status_code == 200
    assert isinstance(load_res.body, dict)
    assert load_res.body.get("success") is True
    _wait_for_model_status(model_id, {"loaded"}, timeout=timeout, headers=headers)


def test_router_unload_model():
    global server
    server.start()
    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"

    _load_model_and_wait(model_id)

    unload_res = server.make_request("POST", "/models/unload", data={"model": model_id})
    assert unload_res.status_code == 200
    assert unload_res.body.get("success") is True
    _wait_for_model_status(model_id, {"unloaded"})


def test_router_unload_force():
    global server
    server.models_max = 1
    server.patience = 30
    server.start()

    model_a = "ggml-org/tinygemma3-GGUF:Q8_0"
    _load_model_and_wait(model_a)

    # Start a request that has some latency
    def keep_busy():
        try:
            server.make_request(
                "POST",
                "/v1/chat/completions",
                data={
                    "model": model_a,
                    "messages": [{"role": "user", "content": "Tell me a long story."}],
                    "max_tokens": 128,
                },
            )
        except Exception:
            pass

    t = threading.Thread(target=keep_busy)
    t.daemon = True
    t.start()

    # Let the request start
    time.sleep(0.5)

    # Force unload. Since patience is 30s and n_requests > 0, a normal unload would block.
    # But a force unload should return immediately and kill the model.
    t_unload_start = time.time()
    unload_res = server.make_request(
        "POST",
        "/models/unload",
        data={"model": model_a, "force": True}
    )
    t_unload_elapsed = time.time() - t_unload_start

    assert unload_res.status_code == 200
    assert unload_res.body.get("success") is True
    assert t_unload_elapsed < 5.0

    _wait_for_model_status(model_a, {"unloaded"})
    t.join(timeout=2)


def test_router_unload_graceful_draining():
    """A graceful unload (force=False) waits for in-flight requests to finish, and incoming requests wait for draining."""
    global server
    server.models_max = 1
    server.start()

    model_a = "ggml-org/tinygemma3-GGUF:Q8_0"
    _load_model_and_wait(model_a)

    in_flight_res = None
    in_flight_err = None

    def run_in_flight():
        nonlocal in_flight_res, in_flight_err
        try:
            in_flight_res = server.make_request(
                "POST",
                "/v1/chat/completions",
                data={
                    "model": model_a,
                    "messages": [{"role": "user", "content": "Tell me a short story."}],
                    "max_tokens": 16,
                },
            )
        except Exception as e:
            in_flight_err = e

    t_flight = threading.Thread(target=run_in_flight)
    t_flight.start()

    # Let the in-flight request start processing
    time.sleep(0.5)

    unload_res = None
    def run_unload():
        nonlocal unload_res
        unload_res = server.make_request(
            "POST",
            "/models/unload",
            data={"model": model_a, "force": False}
        )

    t_unload = threading.Thread(target=run_unload)
    t_unload.start()

    # Give unload thread a moment to transition the model to draining
    time.sleep(0.2)

    # A new request for the draining model must be held by wait_if_draining and succeed after reload
    queued_res = None
    queued_err = None
    def run_queued():
        nonlocal queued_res, queued_err
        try:
            queued_res = server.make_request(
                "POST",
                "/v1/chat/completions",
                data={
                    "model": model_a,
                    "messages": [{"role": "user", "content": "Hello after draining."}],
                    "max_tokens": 4,
                },
                timeout=60,
            )
        except Exception as e:
            queued_err = e

    t_queued = threading.Thread(target=run_queued)
    t_queued.start()

    t_flight.join(timeout=60)
    t_unload.join(timeout=60)
    t_queued.join(timeout=60)

    # In-flight request must complete with 200
    assert in_flight_err is None, f"In-flight request failed: {in_flight_err}"
    assert in_flight_res is not None and in_flight_res.status_code == 200

    # Unload request must succeed with 200
    assert unload_res is not None and unload_res.status_code == 200
    assert unload_res.body.get("success") is True

    # Queued request that arrived during draining must succeed after reload
    assert queued_err is None, f"Queued request during draining failed: {queued_err}"
    assert queued_res is not None and queued_res.status_code == 200


def test_router_models_max_evicts_lru():
    global server
    server.models_max = 2
    server.start()

    candidate_models = [
        "ggml-org/tinygemma3-GGUF:Q8_0",
        "ggml-org/test-model-stories260K:F32",
        "ggml-org/test-model-stories260K-infill:F32",
    ]

    # Load only the first 2 models to fill the cache
    first, second, third = candidate_models[:3]

    _load_model_and_wait(first, timeout=120)
    _load_model_and_wait(second, timeout=120)

    # Verify both models are loaded
    assert _get_model_status(first) == "loaded"
    assert _get_model_status(second) == "loaded"

    # Load the third model - this should trigger LRU eviction of the first model
    _load_model_and_wait(third, timeout=120)

    # Verify eviction: third is loaded, first was evicted
    assert _get_model_status(third) == "loaded"
    assert _get_model_status(first) == "unloaded"


# server_lru_sched tests (relying on LLAMA_SERVER_DEBUG_FAKE_TIMING)

MODEL_A = "ggml-org/tinygemma3-GGUF:Q8_0"
MODEL_B = "ggml-org/test-model-stories260K:F32"
MODEL_C = "ggml-org/test-model-stories260K-infill:F32"


def _tokenize(model_id: str, timeout: float | None = DEFAULT_REQUEST_TIMEOUT) -> ServerResponse:
    return server.make_request(
        "POST", "/tokenize", data={"model": model_id, "content": "hello world"}, timeout=timeout
    )


class _Bg:
    """runs one request in a thread, keeps its result, error and finish time"""

    def __init__(self, fn):
        self.result = None
        self.error: Exception | None = None
        self.done_at: float = 0.0
        self._thread = threading.Thread(target=self._run, args=(fn,), daemon=True)

    def _run(self, fn):
        try:
            self.result = fn()
        except Exception as e:
            self.error = e
        self.done_at = time.time()

    def start(self):
        self._thread.start()
        return self

    def join(self, timeout: int = 180):
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "background request did not finish in time"
        return self

    def assert_ok(self, what: str):
        assert self.error is None, f"{what} raised {self.error!r}"
        assert self.result is not None and self.result.status_code == 200, \
            f"{what} failed: {self.result.status_code if self.result else None} {self.result.body if self.result else None}"


def test_router_queue_does_not_evict_busy_model():
    """a request that finds no free slot waits, and the model serving a request survives it"""
    global server
    server.models_max = 1
    server.start()

    _load_model_and_wait(MODEL_A, timeout=120)

    busy = _Bg(lambda: _tokenize(MODEL_A)).start()
    time.sleep(0.5)  # let the request reach the child and take the only slot

    # no slot free and MODEL_A is busy, so this queues instead of evicting mid-request
    queued = _Bg(lambda: _tokenize(MODEL_B)).start()

    busy.join()
    queued.join()

    # had MODEL_A been evicted while serving, its own request would have died
    busy.assert_ok("request against the busy model")
    queued.assert_ok("queued request")

    _wait_for_model_status(MODEL_B, {"loaded"}, timeout=120)
    assert _get_model_status(MODEL_A) == "unloaded"


def test_router_queue_coalesces_requests_for_same_model():
    """many requests for one missing model share a slot, so only one model is given up"""
    global server
    server.models_max = 2
    server.start()

    _load_model_and_wait(MODEL_A, timeout=120)
    _load_model_and_wait(MODEL_B, timeout=120)

    # keep MODEL_A busy so MODEL_B is the only model that can be given up
    busy = _Bg(lambda: _tokenize(MODEL_A)).start()
    time.sleep(0.5)

    waiters = [_Bg(lambda: _tokenize(MODEL_C)).start() for _ in range(3)]

    busy.join()
    for w in waiters:
        w.join()

    busy.assert_ok("request against the busy model")
    for i, w in enumerate(waiters):
        w.assert_ok(f"queued request {i}")

    _wait_for_model_status(MODEL_C, {"loaded"}, timeout=120)
    # one entry for 3 requests means one eviction: MODEL_B goes, MODEL_A is left alone.
    # without coalescing the leftover entries still ask for a slot,
    # and MODEL_A is taken too as soon as it goes idle
    assert _get_model_status(MODEL_A) == "loaded"
    assert _get_model_status(MODEL_B) == "unloaded"


def test_router_queue_client_disconnect_keeps_model():
    """a client that leaves while queued must not cost a running model its slot"""
    global server
    server.models_max = 1
    server.start()

    _load_model_and_wait(MODEL_A, timeout=120)

    busy = _Bg(lambda: _tokenize(MODEL_A)).start()
    time.sleep(0.5)

    # queues behind MODEL_A, then gives up long before MODEL_A goes idle
    with pytest.raises(requests.exceptions.RequestException):
        _tokenize(MODEL_B, timeout=1)

    busy.join()
    busy.assert_ok("request against the busy model")

    # nobody is waiting anymore, so MODEL_A keeps its slot
    time.sleep(3)
    assert _get_model_status(MODEL_A) == "loaded"
    assert _get_model_status(MODEL_B) == "unloaded"


def test_router_queue_is_fifo():
    """the queue is served in arrival order"""
    global server
    server.models_max = 1
    server.start()

    _load_model_and_wait(MODEL_A, timeout=120)

    busy = _Bg(lambda: _tokenize(MODEL_A)).start()
    time.sleep(0.5)

    first = _Bg(lambda: _tokenize(MODEL_B)).start()
    time.sleep(1)  # keep the arrival order unambiguous
    second = _Bg(lambda: _tokenize(MODEL_C)).start()

    busy.join()
    first.join()
    second.join()

    busy.assert_ok("request against the busy model")
    first.assert_ok("first queued request")
    second.assert_ok("second queued request")

    assert first.done_at < second.done_at, "queue was not served in arrival order"


def test_router_queue_two_waiters_share_one_eviction():
    """two requests that both find the same idle model must both be served in the end"""
    global server
    server.models_max = 1
    server.start()

    _load_model_and_wait(MODEL_A, timeout=120)

    # both arrive while MODEL_A is idle, so both want its slot; only one eviction can happen
    first = _Bg(lambda: _tokenize(MODEL_B)).start()
    second = _Bg(lambda: _tokenize(MODEL_C)).start()

    first.join(90)
    second.join(90)

    first.assert_ok("first queued request")
    second.assert_ok("second queued request")
    assert _get_model_status(MODEL_A) == "unloaded"


def test_router_no_models_autoload():
    global server
    server.no_models_autoload = True
    server.start()
    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"

    res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert res.status_code == 400
    assert "error" in res.body

    _load_model_and_wait(model_id)

    success_res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert success_res.status_code == 200
    assert "error" not in success_res.body


def test_router_api_key_required():
    global server
    server.api_key = "sk-router-secret"
    server.start()

    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"
    auth_headers = {"Authorization": f"Bearer {server.api_key}"}

    res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert res.status_code == 401
    assert res.body.get("error", {}).get("type") == "authentication_error"

    _load_model_and_wait(model_id, headers=auth_headers)

    authed = server.make_request(
        "POST",
        "/v1/chat/completions",
        headers=auth_headers,
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert authed.status_code == 200
    assert "error" not in authed.body


def test_router_reload_models():
    """POST /models/reload re-reads the INI preset and updates the model list."""
    global server

    preset_path = os.path.join(TMP_DIR, "test_reload.ini")

    # Initial preset: two models
    with open(preset_path, "w") as f:
        f.write(
            "[model-reload-a]\n"
            "hf-repo = ggml-org/test-model-stories260K\n"
            "\n"
            "[model-reload-b]\n"
            "hf-repo = ggml-org/test-model-stories260K-infill\n"
        )

    server.models_preset = preset_path
    server.start()

    ids = _get_model_ids(is_reload=False)
    assert "model-reload-a" in ids
    assert "model-reload-b" in ids

    # Updated preset: remove a, keep b unchanged, add c
    with open(preset_path, "w") as f:
        f.write(
            "[model-reload-b]\n"
            "hf-repo = ggml-org/test-model-stories260K-infill\n"
            "\n"
            "[model-reload-c]\n"
            "hf-repo = ggml-org/test-model-stories260K\n"
        )

    try:
        ids = _get_model_ids(is_reload=True)
        assert "model-reload-a" not in ids, "removed model should no longer appear"
        assert "model-reload-b" in ids, "unchanged model should still appear"
        assert "model-reload-c" in ids, "newly added model should appear"
    finally:
        os.remove(preset_path)


def test_router_dedup_cache_models():
    """dedup-cache-models hides the cache entry backing a preset from GET /models"""
    global server

    preset_path = os.path.join(TMP_DIR, "test_dedup.ini")
    cache_id = "ggml-org/test-model-stories260K:F32"

    with open(preset_path, "w") as f:
        f.write(
            "[model-dedup]\n"
            "hf-repo = ggml-org/test-model-stories260K\n"
            "dedup-cache-models = 1\n"
        )

    server.models_preset = preset_path
    server.start()

    try:
        ids = _get_model_ids(is_reload=False)
        assert "model-dedup" in ids
        assert cache_id not in ids, "cache model should be hidden by dedup"
        # other cache models are unaffected
        assert "ggml-org/tinygemma3-GGUF:Q8_0" in ids

        # the hidden model is only hidden from the listing, it can still be used
        res = server.make_request("POST", "/tokenize", data={"model": cache_id, "content": "hello"})
        assert res.status_code == 200

        # disabling the flag brings the cache entry back on reload
        with open(preset_path, "w") as f:
            f.write(
                "[model-dedup]\n"
                "hf-repo = ggml-org/test-model-stories260K\n"
            )
        ids = _get_model_ids(is_reload=True)
        assert cache_id in ids

        # the flag also works from the global section
        with open(preset_path, "w") as f:
            f.write(
                "[*]\n"
                "dedup-cache-models = 1\n"
                "\n"
                "[model-dedup]\n"
                "hf-repo = ggml-org/test-model-stories260K\n"
            )
        ids = _get_model_ids(is_reload=True)
        assert "model-dedup" in ids
        assert cache_id not in ids, "cache model should be hidden by global dedup"
    finally:
        os.remove(preset_path)


def test_router_remote_preset():
    global server
    server.model_hf_repo = "ggml-org/test-preset-ci"
    server.model_hf_file = None
    server.offline = False
    server.start()

    # Should see preset models in GET /models
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    ids = {item["id"] for item in res.body.get("data", [])}
    assert "tinygemma3-preset" in ids
    assert "stories260K-test" in ids

    # Should be able to load a preset model
    model_id = "tinygemma3-preset"
    _load_model_and_wait(model_id)


MODEL_DOWNLOAD_ID = "ggml-org/test-model-router-download:F16"
MODEL_DOWNLOAD_TIMEOUT = 30


def _listen_sse(
    server: ServerProcess, collected: list, stop: threading.Event, ready: threading.Event | None = None
):
    """Collect /models/sse events into `collected` until `stop` is set.

    When `ready` is provided, it is set once the streaming response is open,
    i.e. the server has accepted the connection and registered us as a
    subscriber. Callers that trigger one-shot events (e.g. download_finished)
    must wait on `ready` before acting, otherwise the event can be broadcast
    before this client is subscribed and be lost.
    """
    url = f"http://{server.server_host}:{server.server_port}/models/sse"
    try:
        with requests.get(url, stream=True, timeout=MODEL_DOWNLOAD_TIMEOUT) as resp:
            if ready is not None:
                ready.set()
            for line_bytes in resp.iter_lines():
                if stop.is_set():
                    break
                line = line_bytes.decode("utf-8")
                if line.startswith("data: "):
                    collected.append(json.loads(line[6:]))
    except Exception:
        pass


def _wait_for_sse_event(collected: list, event_type: str, model: str, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if any(e.get("event") == event_type and e.get("model") == model for e in collected):
            return True
        time.sleep(0.01)
    return False


def test_router_download_model():
    """Case 1: download a model, verify SSE events and GET /models."""
    global server
    server.start()

    # Ensure the model is not present before we start
    server.make_request("DELETE", f"/models?model={MODEL_DOWNLOAD_ID}")

    sse_events: list = []
    stop = threading.Event()
    sse_ready = threading.Event()
    sse_thread = threading.Thread(
        target=_listen_sse, args=(server, sse_events, stop, sse_ready), daemon=True
    )
    sse_thread.start()

    # wait for the SSE client to be subscribed before triggering the download,
    # otherwise the one-shot download_finished event can be broadcast before
    # this client is registered and be lost
    assert sse_ready.wait(10), "SSE client failed to connect"

    # Trigger the download
    res = server.make_request("POST", "/models", data={"model": MODEL_DOWNLOAD_ID})
    assert res.status_code == 200
    assert res.body.get("success") is True

    # Wait for download_finished SSE event
    finished = _wait_for_sse_event(
        sse_events, "download_finished", MODEL_DOWNLOAD_ID, MODEL_DOWNLOAD_TIMEOUT
    )
    stop.set()

    assert finished, "Never received download_finished SSE event"
    assert any(
        e.get("event") == "download_progress" and e.get("model") == MODEL_DOWNLOAD_ID
        for e in sse_events
    ), "No download_progress events received"

    # Model should now appear in GET /models
    ids = _get_model_ids(is_reload=False)
    assert MODEL_DOWNLOAD_ID in ids, f"{MODEL_DOWNLOAD_ID} not found in /models after download"


def test_router_delete_model():
    """Case 2: delete the downloaded model, verify it disappears from GET /models."""
    global server
    server.start()

    # Ensure the model exists (download it if needed)
    if MODEL_DOWNLOAD_ID not in _get_model_ids(is_reload=False):
        sse_events: list = []
        stop = threading.Event()
        sse_ready = threading.Event()
        threading.Thread(
            target=_listen_sse, args=(server, sse_events, stop, sse_ready), daemon=True
        ).start()
        # subscribe before triggering the download so the one-shot
        # download_finished event is not lost (see test_router_download_model)
        assert sse_ready.wait(10), "SSE client failed to connect"
        res = server.make_request("POST", "/models", data={"model": MODEL_DOWNLOAD_ID})
        assert res.status_code == 200
        finished = _wait_for_sse_event(
            sse_events, "download_finished", MODEL_DOWNLOAD_ID, MODEL_DOWNLOAD_TIMEOUT
        )
        stop.set()
        assert finished, "Model did not finish downloading before delete test"

    # Delete the model
    del_res = server.make_request("DELETE", f"/models?model={MODEL_DOWNLOAD_ID}")
    assert del_res.status_code == 200
    assert del_res.body.get("success") is True

    # Model should no longer appear in GET /models
    ids = _get_model_ids(is_reload=False)
    assert MODEL_DOWNLOAD_ID not in ids, f"{MODEL_DOWNLOAD_ID} still present after deletion"


def test_router_queues_swapping_requests():
    global server
    server.models_max = 1
    server.start()

    model_a = "ggml-org/tinygemma3-GGUF:Q8_0"
    model_b = "ggml-org/test-model-stories260K:F32"

    _load_model_and_wait(model_a)

    def run_model_a():
        return server.make_request(
            "POST",
            "/v1/chat/completions",
            data={
                "model": model_a,
                "messages": [{"role": "user", "content": "Write a long story about a happy cat."}],
                "max_tokens": 16,
            },
        )

    def run_model_b():
        time.sleep(0.5) # ensure model A's request starts first
        return server.make_request(
            "POST",
            "/v1/chat/completions",
            data={
                "model": model_b,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 4,
            },
        )

    results = parallel_function_calls([
        (run_model_a, ()),
        (run_model_b, ()),
    ])

    res_a, res_b = results
    assert res_a.status_code == 200
    assert res_b.status_code == 200
    assert "error" not in res_a.body
    assert "error" not in res_b.body


def test_router_patience_window():
    global server
    server.models_max = 1
    server.patience = 5  # 5 seconds patience
    server.start()

    model_a = "ggml-org/tinygemma3-GGUF:Q8_0"
    model_b = "ggml-org/test-model-stories260K:F32"

    _load_model_and_wait(model_a)

    # Sequence 1: patience extension
    stop_event = threading.Event()
    threads = []
    
    def keep_busy():
        while not stop_event.is_set():
            try:
                server.make_request(
                    "POST",
                    "/v1/chat/completions",
                    data={
                        "model": model_a,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 16,
                    },
                )
            except Exception:
                pass
            time.sleep(0.01)

    # Start 3 background threads to keep model_a constantly busy
    for _ in range(3):
        t = threading.Thread(target=keep_busy)
        t.daemon = True
        t.start()
        threads.append(t)

    # Let the background threads run for a moment to ensure model_a has requests
    time.sleep(0.5)

    def run_request_b():
        t_start = time.time()
        res = server.make_request(
            "POST",
            "/v1/chat/completions",
            data={
                "model": model_b,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 4,
            },
        )
        return res, time.time() - t_start

    def run_request_a2():
        # A2 starts 1.5s after B started (so at t = 2.0s overall)
        time.sleep(1.5)
        t_start = time.time()
        res = server.make_request(
            "POST",
            "/v1/chat/completions",
            data={
                "model": model_a,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
            },
        )
        return res, time.time() - t_start

    def stop_busy_loop():
        # Stop the background threads 3.5s after B started (so at t = 4.0s overall)
        time.sleep(3.5)
        stop_event.set()
        for t in threads:
            t.join()
        return None

    results = parallel_function_calls([
        (run_request_b, ()),
        (run_request_a2, ()),
        (stop_busy_loop, ()),
    ])

    (res_b, b_duration), (res_a2, a2_duration), _ = results
    assert res_b.status_code == 200
    assert res_a2.status_code == 200
    assert "error" not in res_a2.body

    # A2 should have run immediately (short duration, not waiting for swap)
    assert a2_duration < 3.0
    # B should have blocked waiting for patience timeout + A2 to complete
    assert b_duration >= 3.5

    # Sequence 2: voluntarily idle unload
    # Reload model_a to trigger the idle unload test
    _load_model_and_wait(model_a)

    # Set server patience to a larger value (15s) and start requests.
    # Request A completes quickly. Request B starts at t=0.5s.
    # Once A completes, A's active request count drops to 0, so it should unload immediately.
    server.stop()
    server.patience = 15
    server.start()
    _load_model_and_wait(model_a)

    def run_quick_a():
        return server.make_request(
            "POST",
            "/v1/chat/completions",
            data={
                "model": model_a,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 2,
            },
        )

    def run_pending_b():
        time.sleep(0.5)
        t_start = time.time()
        res = server.make_request(
            "POST",
            "/v1/chat/completions",
            data={
                "model": model_b,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 4,
            },
        )
        return res, time.time() - t_start

    results2 = parallel_function_calls([
        (run_quick_a, ()),
        (run_pending_b, ()),
    ])

    res_quick_a, (res_pending_b, b2_duration) = results2
    assert res_quick_a.status_code == 200
    assert res_pending_b.status_code == 200
    # B should load quickly, way before the 15-second patience timeout, because A became idle.
    assert b2_duration < 8.0


def test_router_bounded_queue():
    global server
    server.models_max = 1
    server.max_waiting_requests = 1
    server.start()

    model_a = "ggml-org/tinygemma3-GGUF:Q8_0"
    model_b = "ggml-org/test-model-stories260K:F32"
    model_c = "ggml-org/test-model-stories260K-infill:F32"

    _load_model_and_wait(model_a)

    stop_event = threading.Event()
    threads = []
    
    def keep_busy():
        while not stop_event.is_set():
            try:
                server.make_request(
                    "POST",
                    "/v1/chat/completions",
                    data={
                        "model": model_a,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 16,
                    },
                )
            except Exception:
                pass
            time.sleep(0.01)

    # Start 3 background threads to keep model_a constantly busy
    for _ in range(3):
        t = threading.Thread(target=keep_busy)
        t.daemon = True
        t.start()
        threads.append(t)

    # Let the background threads run to ensure model_a has requests
    time.sleep(0.5)

    def run_model_b_queued():
        t_start = time.time()
        res = server.make_request(
            "POST",
            "/v1/chat/completions",
            data={
                "model": model_b,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 4,
            },
        )
        return res, time.time() - t_start

    def run_model_c_rejected():
        time.sleep(0.5)
        t_start = time.time()
        res = server.make_request(
            "POST",
            "/v1/chat/completions",
            data={
                "model": model_c,
                "messages": [{"role": "user", "content": "hello infill"}],
                "max_tokens": 4,
            },
        )
        return res, time.time() - t_start

    def stop_busy_loop():
        time.sleep(2.0)
        stop_event.set()
        for t in threads:
            t.join()
        return None

    results = parallel_function_calls([
        (run_model_b_queued, ()),
        (run_model_c_rejected, ()),
        (stop_busy_loop, ()),
    ])

    (res_b, b_duration), (res_c, c_duration), _ = results
    assert res_b.status_code == 200
    assert res_c.status_code in [429, 503]
    if res_c.status_code == 429:
        retry_after = res_c.headers.get("retry-after") or res_c.headers.get("Retry-After")
        assert retry_after is not None, "429 response missing Retry-After header"
        assert int(retry_after) >= 1, f"Retry-After header value invalid: {retry_after}"


def test_router_concurrent_load():
    import os
    global server
    server.models_max = 1
    # Create a unique log path for this test so we can inspect it
    server.log_path = "tmp/test_router_concurrent_load.log"
    if os.path.exists(server.log_path):
        os.remove(server.log_path)
    server.start()

    model_a = "ggml-org/tinygemma3-GGUF:Q8_0"

    def load_req():
        try:
            return server.make_request("POST", "/models/load", data={"model": model_a})
        except Exception as e:
            return e

    # Send 2 load requests concurrently
    results = parallel_function_calls([
        (load_req, ()),
        (load_req, ()),
    ])

    # Both requests should either return success (200) or report that the model is already running (400)
    for res in results:
        assert not isinstance(res, Exception), f"Request failed with exception: {res}"
        if res.status_code == 400:
            assert "already running" in res.body.get("error", {}).get("message", "")
        else:
            assert res.status_code == 200
            assert res.body.get("success") is True

    # Allow some time for processes/threads to stabilize and logs to flush
    time.sleep(1.0)

    # Stop the server to release the log file
    server.stop()

    # Read log and verify only one instance was spawned, and no "old process... still alive" warning
    assert os.path.exists(server.log_path)
    with open(server.log_path, "r") as f:
        log_content = f.read()

    # The router should print the spawn message exactly once for this model
    spawn_count = log_content.count("spawning server instance with name=ggml-org/tinygemma3-GGUF:Q8_0")
    assert spawn_count == 1, f"Expected exactly 1 spawn log, found {spawn_count}"

    # We should NOT see the warning about old process still alive
    assert "old process for model name=ggml-org/tinygemma3-GGUF:Q8_0 is still alive" not in log_content


def test_router_patience_evicts_busy_model():
    """When patience expires under sustained traffic, the LRU busy model is evicted to free a slot."""
    global server
    server.models_max = 1
    server.patience = 2  # 2s patience
    server.start()

    model_a = "ggml-org/tinygemma3-GGUF:Q8_0"
    model_b = "ggml-org/test-model-stories260K:F32"

    _load_model_and_wait(model_a)

    stop_event = threading.Event()
    threads = []

    def keep_busy():
        while not stop_event.is_set():
            if _get_model_status(model_a) != "loaded":
                break
            try:
                server.make_request(
                    "POST",
                    "/v1/chat/completions",
                    data={
                        "model": model_a,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 8,
                    },
                )
            except Exception:
                pass
            time.sleep(0.05)

    for _ in range(2):
        t = threading.Thread(target=keep_busy)
        t.daemon = True
        t.start()
        threads.append(t)
    time.sleep(0.5)

    def run_model_b():
        t_start = time.time()
        try:
            res = server.make_request(
                "POST",
                "/v1/chat/completions",
                data={
                    "model": model_b,
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 4,
                },
                timeout=60,
            )
            return res, time.time() - t_start
        finally:
            stop_event.set()

    def stop_busy_loop():
        time.sleep(2.5)
        stop_event.set()
        return None

    results = parallel_function_calls([
        (run_model_b, ()),
        (stop_busy_loop, ()),
    ])

    for t in threads:
        t.join(timeout=5)

    (res_b, elapsed), _ = results

    assert res_b.status_code == 200
    assert "error" not in res_b.body
    # Should have waited at least the 2s patience window
    assert elapsed >= 2.0
    # Model B is now loaded, Model A was evicted
    assert _get_model_status(model_b) == "loaded"
    assert _get_model_status(model_a) == "unloaded"


