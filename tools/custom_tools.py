"""Custom tools for the Flaky Test Investigator crew."""

from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Tool 1 – Test History Tool
# ──────────────────────────────────────────────

class TestHistoryToolInput(BaseModel):
    """Input schema for TestHistoryTool."""
    query: str = Field(
        default="all",
        description="Optional filter query. Use 'all' to retrieve history for every test.",
    )


class TestHistoryTool(BaseTool):
    name: str = "Test History Tool"
    description: str = (
        "Retrieves the pass/fail history for the last 10 runs of every test "
        "in the suite. Returns a dictionary where each key is a test name and "
        "the value is a list of 10 booleans (True = pass, False = fail). "
        "Pass query='all' or leave it as default."
    )
    args_schema: Type[BaseModel] = TestHistoryToolInput

    def _run(self, query: str = "all") -> dict:
        """Return mock test-history data."""
        return {
            # ── Stable tests (all pass) ──
            "test_login_valid_credentials": [True, True, True, True, True, True, True, True, True, True],
            "test_logout_success": [True, True, True, True, True, True, True, True, True, True],
            "test_homepage_loads": [True, True, True, True, True, True, True, True, True, True],
            "test_add_item_to_cart": [True, True, True, True, True, True, True, True, True, True],

            # ── Flaky tests (mixed results) ──
            "test_payment_processing": [True, False, True, True, False, True, False, True, True, False],
            "test_search_results_order": [True, True, False, True, False, False, True, True, False, True],
            "test_dashboard_widget_render": [False, True, True, False, True, False, True, False, True, True],
            "test_email_notification_sent": [True, False, False, True, True, False, True, False, True, False],
            "test_concurrent_file_upload": [False, True, False, True, False, True, False, True, False, True],
            "test_realtime_chat_message": [True, False, True, False, True, False, True, False, True, False],
        }


# ──────────────────────────────────────────────
# Tool 2 – Test Source Code Tool
# ──────────────────────────────────────────────

class TestSourceCodeToolInput(BaseModel):
    """Input schema for TestSourceCodeTool."""
    test_name: str = Field(
        ...,
        description="The name of the test whose source code should be retrieved.",
    )


class TestSourceCodeTool(BaseTool):
    name: str = "Test Source Code Tool"
    description: str = (
        "Retrieves the source code of a given test by name. "
        "Pass the exact test function name (e.g. 'test_payment_processing')."
    )
    args_schema: Type[BaseModel] = TestSourceCodeToolInput

    def _run(self, test_name: str) -> str:
        """Return mock source code for the requested test."""
        source_code_map = {
            # ── Stable tests ──
            "test_login_valid_credentials": '''
def test_login_valid_credentials(client):
    """Stable test – deterministic assertions."""
    response = client.post("/login", json={"user": "admin", "password": "secret"})
    assert response.status_code == 200
    assert response.json()["token"] is not None
''',
            "test_logout_success": '''
def test_logout_success(client, auth_token):
    """Stable test – straightforward logout."""
    response = client.post("/logout", headers={"Authorization": f"Bearer {auth_token}"})
    assert response.status_code == 200
''',
            "test_homepage_loads": '''
def test_homepage_loads(client):
    """Stable test – simple GET request."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.text
''',
            "test_add_item_to_cart": '''
def test_add_item_to_cart(client, auth_token):
    """Stable test – cart manipulation."""
    response = client.post("/cart", json={"item_id": 42, "qty": 1},
                           headers={"Authorization": f"Bearer {auth_token}"})
    assert response.status_code == 201
    assert response.json()["cart_size"] == 1
''',

            # ── Flaky tests ──
            "test_payment_processing": '''
import time

def test_payment_processing(client, auth_token):
    """FLAKY – uses time.sleep() to wait for payment gateway callback."""
    client.post("/pay", json={"amount": 99.99},
                headers={"Authorization": f"Bearer {auth_token}"})
    time.sleep(3)  # arbitrary wait for async callback
    status = client.get("/pay/status", headers={"Authorization": f"Bearer {auth_token}"})
    assert status.json()["paid"] is True
''',
            "test_search_results_order": '''
def test_search_results_order(client):
    """FLAKY – relies on non-deterministic ordering from the database."""
    response = client.get("/search?q=shoes")
    results = response.json()["items"]
    # Assumes a specific order that the DB does not guarantee
    assert results[0]["name"] == "Running Shoes"
    assert results[1]["name"] == "Casual Shoes"
''',
            "test_dashboard_widget_render": '''
import threading

def test_dashboard_widget_render(client, auth_token):
    """FLAKY – spawns background thread for widget data hydration."""
    data = {}

    def fetch_widget():
        resp = client.get("/widgets", headers={"Authorization": f"Bearer {auth_token}"})
        data["widgets"] = resp.json()

    t = threading.Thread(target=fetch_widget)
    t.start()
    t.join(timeout=2)
    assert len(data.get("widgets", [])) > 0
''',
            "test_email_notification_sent": '''
import time

def test_email_notification_sent(client, auth_token):
    """FLAKY – polls an external email service with a hard-coded sleep."""
    client.post("/notify", json={"to": "user@example.com", "msg": "Hi"},
                headers={"Authorization": f"Bearer {auth_token}"})
    time.sleep(5)  # waiting for email delivery
    emails = client.get("/emails?to=user@example.com")
    assert any(e["subject"] == "Hi" for e in emails.json())
''',
            "test_concurrent_file_upload": '''
import threading
import time

def test_concurrent_file_upload(client, auth_token):
    """FLAKY – concurrent uploads with race condition on shared state."""
    results = []

    def upload(file_name):
        time.sleep(0.1)
        resp = client.post("/upload", files={"file": (file_name, b"data")},
                           headers={"Authorization": f"Bearer {auth_token}"})
        results.append(resp.status_code)

    threads = [threading.Thread(target=upload, args=(f"file_{i}.txt",)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r == 201 for r in results)
''',
            "test_realtime_chat_message": '''
import time
import threading

def test_realtime_chat_message(client, auth_token):
    """FLAKY – depends on WebSocket timing and shared mutable state."""
    received = []

    def listen():
        ws = client.websocket_connect("/ws/chat")
        msg = ws.receive_json(timeout=3)
        received.append(msg)

    listener = threading.Thread(target=listen)
    listener.start()
    time.sleep(1)
    client.post("/chat/send", json={"text": "hello"},
                headers={"Authorization": f"Bearer {auth_token}"})
    listener.join(timeout=5)
    assert len(received) == 1
    assert received[0]["text"] == "hello"
''',
        }

        if test_name in source_code_map:
            return source_code_map[test_name]
        return f"ERROR: No source code found for test '{test_name}'."
