"""Tests for append-only /v1/session-events ingestion."""

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


class FakeRequest:
    def __init__(self, body=None, headers=None):
        self._body = body
        self.headers = headers or {}

    async def json(self):
        if isinstance(self._body, BaseException):
            raise self._body
        return self._body


async def _post_session_event(adapter: APIServerAdapter, body, headers=None):
    return await adapter._handle_session_events(FakeRequest(body, headers=headers))


def _response_json(response):
    import json

    return json.loads(response.text)


@pytest.fixture()
def session_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    yield db
    db.close()


@pytest.mark.asyncio
async def test_session_event_requires_auth_when_key_configured(session_db):
    adapter = _make_adapter(api_key="sk-secret")
    adapter._session_db = session_db
    resp = await _post_session_event(
        adapter,
        {
            "session_key": "voice:CA123",
            "event_type": "user.transcript.final",
            "content": "hello",
        },
    )

    assert resp.status == 401


@pytest.mark.asyncio
async def test_session_event_rejects_invalid_event_type(session_db):
    adapter = _make_adapter()
    adapter._session_db = session_db
    resp = await _post_session_event(
        adapter,
        {
            "session_key": "voice:CA123",
            "event_type": "partial.transcript",
            "content": "hello",
        },
    )

    assert resp.status == 400
    data = _response_json(resp)
    assert data["error"]["code"] == "invalid_event_type"


@pytest.mark.asyncio
async def test_session_event_appends_final_user_transcript(session_db):
    adapter = _make_adapter(api_key="sk-secret")
    adapter._session_db = session_db
    resp = await _post_session_event(
        adapter,
        {
            "session_key": "voice:CA123",
            "call_sid": "CA123",
            "event_type": "user.transcript.final",
            "content": "turn on the office lights",
        },
        headers={"Authorization": "Bearer sk-secret"},
    )
    data = _response_json(resp)

    assert resp.status == 200
    assert data["appended"] is True
    session_id = data["session_id"]
    assert resp.headers["X-Hermes-Session-Id"] == session_id

    session = session_db.get_session(session_id)
    assert session["source"] == "voice"
    assert session["user_id"] == "CA123"

    messages = session_db.get_messages_as_conversation(session_id)
    assert messages == [{"role": "user", "content": "turn on the office lights"}]


@pytest.mark.asyncio
async def test_session_event_can_append_tool_result(session_db):
    adapter = _make_adapter()
    adapter._session_db = session_db
    resp = await _post_session_event(
        adapter,
        {
            "session_key": "voice:CA123",
            "session_id": "voice-call-123",
            "event_type": "tool.completed",
            "role": "tool",
            "tool_name": "remember_fact",
            "tool_call_id": "call_1",
            "content": "accepted",
        },
    )
    data = _response_json(resp)

    assert resp.status == 200
    assert data["session_id"] == "voice-call-123"
    messages = session_db.get_messages_as_conversation("voice-call-123")
    assert messages[0]["role"] == "tool"
    assert messages[0]["content"] == "accepted"
    assert messages[0]["tool_name"] == "remember_fact"
    assert messages[0]["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_call_ended_marks_session_ended_without_message(session_db):
    adapter = _make_adapter()
    adapter._session_db = session_db
    resp = await _post_session_event(
        adapter,
        {
            "session_key": "voice:CA123",
            "session_id": "voice-call-123",
            "event_type": "call.ended",
        },
    )
    data = _response_json(resp)

    assert resp.status == 200
    assert data["appended"] is False
    session = session_db.get_session("voice-call-123")
    assert session["end_reason"] == "voice_call_ended"
    assert session_db.get_messages_as_conversation("voice-call-123") == []


@pytest.mark.asyncio
async def test_capabilities_advertises_session_events(session_db):
    adapter = _make_adapter()
    adapter._session_db = session_db
    resp = await adapter._handle_capabilities(FakeRequest(headers={}))
    data = _response_json(resp)

    assert resp.status == 200
    assert data["features"]["session_event_ingest"] is True
    assert data["endpoints"]["session_events"] == {
        "method": "POST",
        "path": "/v1/session-events",
    }
