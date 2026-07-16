#!/usr/bin/env python3
"""Tests for scripts/9router_tokscale_bridge_gjc.py.

Run with: python3 -m pytest scripts/test_9router_tokscale_bridge_gjc.py

The bridge module's filename starts with a digit ("9router_..."), which is
not a valid Python identifier, so it can't be imported with a plain
`import` statement. It is loaded here via importlib.util instead.
"""

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).parent / "9router_tokscale_bridge_gjc.py"
spec = importlib.util.spec_from_file_location("bridge_9router", MODULE_PATH)
bridge = importlib.util.module_from_spec(spec)
sys.modules["bridge_9router"] = bridge
spec.loader.exec_module(bridge)


def make_row(**overrides):
    """Build a dict-based fake DB row with sensible defaults.

    `convert_row_to_entry` only ever accesses rows via `row["col"]`
    (bracket access), which a plain dict supports the same way
    `sqlite3.Row` does, so dicts are a valid stand-in in tests.
    """
    row = {
        "id": "req_1",
        "timestamp": "2026-01-15T12:00:00Z",
        "provider": "openai",
        "model": "gpt-4o",
        "connectionId": "conn_1",
        "data": json.dumps(
            {
                "model": "gpt-4o",
                "tokens": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "cached_tokens": 20,
                },
            }
        ),
    }
    row.update(overrides)
    return row


# ── compute_token_buckets: OpenAI prompt-includes-cached split ─────────────


def test_compute_token_buckets_splits_prompt_and_cached():
    # prompt_tokens (100) already includes cached_tokens (20), so the
    # non-cached input bucket is prompt - cached = 80, and total sums the
    # three non-overlapping buckets back up.
    input_tokens, cached, completion, total = bridge.compute_token_buckets(
        {"prompt_tokens": 100, "completion_tokens": 50, "cached_tokens": 20}
    )
    assert input_tokens == 80
    assert cached == 20
    assert completion == 50
    assert total == input_tokens + cached + completion
    assert total == 150


def test_compute_token_buckets_no_cache():
    input_tokens, cached, completion, total = bridge.compute_token_buckets(
        {"prompt_tokens": 100, "completion_tokens": 50}
    )
    assert input_tokens == 100
    assert cached == 0
    assert completion == 50
    assert total == 150


def test_compute_token_buckets_cached_greater_than_prompt_clamps_to_zero():
    # Malformed/unexpected upstream data where cached_tokens exceeds
    # prompt_tokens must not go negative.
    input_tokens, cached, completion, total = bridge.compute_token_buckets(
        {"prompt_tokens": 10, "completion_tokens": 5, "cached_tokens": 999}
    )
    assert input_tokens == 0
    assert cached == 999
    assert completion == 5
    assert total == 0 + 999 + 5


# ── convert_row_to_entry: end-to-end token math via a full row ─────────────


def test_convert_row_to_entry_computes_non_overlapping_usage():
    row = make_row()
    result = bridge.convert_row_to_entry(row)
    assert result is not None
    usage = result["entry"]["message"]["usage"]
    assert usage["input"] == 80
    assert usage["cacheRead"] == 20
    assert usage["output"] == 50
    assert usage["totalTokens"] == 150


# ── malformed / NULL `data` JSON rows are skipped ───────────────────────────


def test_convert_row_to_entry_skips_null_data():
    row = make_row(data=None)
    assert bridge.convert_row_to_entry(row) is None


def test_convert_row_to_entry_skips_malformed_json_data():
    row = make_row(data="{not valid json")
    assert bridge.convert_row_to_entry(row) is None


def test_convert_row_to_entry_skips_non_dict_data():
    row = make_row(data=json.dumps([1, 2, 3]))
    assert bridge.convert_row_to_entry(row) is None


# ── non-dict `tokens` field is skipped ──────────────────────────────────────


def test_convert_row_to_entry_skips_non_dict_tokens():
    row = make_row(data=json.dumps({"model": "gpt-4o", "tokens": [1, 2, 3]}))
    assert bridge.convert_row_to_entry(row) is None


def test_convert_row_to_entry_skips_string_tokens():
    row = make_row(data=json.dumps({"model": "gpt-4o", "tokens": "bogus"}))
    assert bridge.convert_row_to_entry(row) is None


def test_convert_row_to_entry_skips_zero_prompt_and_completion():
    row = make_row(
        data=json.dumps(
            {
                "model": "gpt-4o",
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0},
            }
        )
    )
    assert bridge.convert_row_to_entry(row) is None


# ── local-date bucketing ────────────────────────────────────────────────────


def test_convert_row_to_entry_buckets_by_local_date():
    # ts_ms is derived from the ISO timestamp, then the bucket date is
    # computed via .astimezone() (local time), not UTC. Compare against the
    # same conversion tokscale's bridge performs so the test tracks the
    # implementation's documented local-date behavior rather than hardcoding
    # a timezone-dependent literal.
    from datetime import datetime, timezone

    row = make_row(timestamp="2026-01-15T23:30:00Z")
    result = bridge.convert_row_to_entry(row)
    assert result is not None

    ts_ms = result["entry"]["message"]["timestamp"]
    expected_date = (
        datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%d")
    )
    assert result["date_str"] == expected_date


def test_convert_row_to_entry_groups_same_utc_day_rows_into_one_bucket():
    row_a = make_row(id="req_a", timestamp="2026-01-15T01:00:00Z")
    row_b = make_row(id="req_b", timestamp="2026-01-15T02:00:00Z")
    result_a = bridge.convert_row_to_entry(row_a)
    result_b = bridge.convert_row_to_entry(row_b)
    assert result_a["date_str"] == result_b["date_str"]


# ── source stamping ─────────────────────────────────────────────────────────


def test_convert_row_to_entry_stamps_9router_source():
    row = make_row()
    result = bridge.convert_row_to_entry(row)
    assert result["entry"]["message"]["source"] == "9router"


# ── atomic write cleans up its .tmp file on failure ─────────────────────────


def test_run_cleans_up_tmp_file_when_write_fails(tmp_path, monkeypatch):
    db_path = tmp_path / "data.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE requestDetails (id TEXT, timestamp TEXT, provider TEXT, "
        "model TEXT, connectionId TEXT, data TEXT, status TEXT)"
    )
    conn.execute(
        "INSERT INTO requestDetails VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "req_1",
            "2026-01-15T12:00:00Z",
            "openai",
            "gpt-4o",
            "conn_1",
            json.dumps(
                {
                    "model": "gpt-4o",
                    "tokens": {"prompt_tokens": 10, "completion_tokens": 5, "cached_tokens": 0},
                }
            ),
            "success",
        ),
    )
    conn.commit()
    conn.close()

    bridge_dir = tmp_path / "sessions"
    monkeypatch.setattr(bridge, "ROUTER_DB", db_path)
    monkeypatch.setattr(bridge, "BRIDGE_DIR", bridge_dir)

    original_dumps = bridge.json.dumps
    call_count = {"n": 0}

    def flaky_dumps(*args, **kwargs):
        call_count["n"] += 1
        # Let the session header line through, then blow up on the first
        # message line so the write fails mid-file, after the .tmp file
        # was created but before os.replace() runs.
        if call_count["n"] == 2:
            raise RuntimeError("boom")
        return original_dumps(*args, **kwargs)

    monkeypatch.setattr(bridge.json, "dumps", flaky_dumps)

    try:
        bridge.run()
        assert False, "expected run() to propagate the write failure"
    except RuntimeError:
        pass

    leftover_tmp_files = list(bridge_dir.glob("*.tmp"))
    assert leftover_tmp_files == [], f"leaked .tmp files: {leftover_tmp_files}"
