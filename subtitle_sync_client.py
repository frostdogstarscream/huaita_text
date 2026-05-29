from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class SubtitleSyncError(Exception):
    pass


class SubtitleSyncUnavailableError(SubtitleSyncError):
    pass


class SubtitlePlaylistMismatchError(SubtitleSyncError):
    pass


class SubtitleStateValidationError(SubtitleSyncError):
    pass


@dataclass
class SubtitleStateCache:
    playlist_id: str
    slide_count: int
    sequence_no: int
    slide_id: str
    interval_seconds: float
    playing: bool
    revision: int
    changed_at: str
    received_at: float
    source: str = "remote"


class SubtitleSyncClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self._base_url = config.get("base_url", "http://127.0.0.1:10061").rstrip("/")
        self._expected_playlist_id = config.get("expected_playlist_id", "huaihai-75-v1")
        self._expected_slide_count = int(config.get("expected_slide_count", 75))
        self._request_timeout = float(config.get("request_timeout_seconds", 0.5))
        self._max_cached_age = float(config.get("max_cached_age_seconds", 1.0))
        self._lock = threading.Lock()
        self._cache: SubtitleStateCache | None = None
        self._last_success_at: float = 0.0
        self._last_error: str = ""
        self._connected: bool = False

    def refresh(self) -> SubtitleStateCache:
        state = self._request_state()
        if not state.get("ok"):
            raise SubtitleSyncUnavailableError("Projection server reports not ok")
        self._validate_state(state)
        cache = SubtitleStateCache(
            playlist_id=state["playlist_id"],
            slide_count=int(state["slide_count"]),
            sequence_no=int(state["sequence_no"]),
            slide_id=state.get("slide_id", f"slide_{int(state['sequence_no']):03d}"),
            interval_seconds=float(state.get("interval_seconds", 5)),
            playing=bool(state.get("playing", True)),
            revision=int(state.get("revision", 0)),
            changed_at=state.get("changed_at", ""),
            received_at=time.time(),
            source="remote",
        )
        with self._lock:
            if self._cache is None or cache.revision >= self._cache.revision:
                self._cache = cache
            self._last_success_at = cache.received_at
            self._last_error = ""
            self._connected = True
        return cache

    def refresh_or_raise(self) -> SubtitleStateCache:
        try:
            return self.refresh()
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
                self._connected = False
            raise

    def get_cached_state(self, max_age_seconds: float | None = None) -> SubtitleStateCache | None:
        age = max_age_seconds if max_age_seconds is not None else self._max_cached_age
        with self._lock:
            if self._cache is None:
                return None
            if time.time() - self._cache.received_at > age:
                return None
            return self._cache

    def resolve_state_for_capture(self) -> SubtitleStateCache:
        try:
            return self.refresh()
        except (SubtitlePlaylistMismatchError, SubtitleStateValidationError):
            raise
        except SubtitleSyncUnavailableError:
            cached = self.get_cached_state()
            if cached is not None:
                cached.source = "cached_remote"
                return cached
            raise

    def status(self) -> dict[str, Any]:
        with self._lock:
            cached = self._cache
            return {
                "enabled": True,
                "connected": self._connected,
                "sequence_no": cached.sequence_no if cached else None,
                "revision": cached.revision if cached else None,
                "last_success_at": self._last_success_at,
                "last_error": self._last_error,
                "cache_age_seconds": time.time() - self._last_success_at if self._last_success_at else None,
                "base_url": self._base_url,
                "expected_playlist_id": self._expected_playlist_id,
                "expected_slide_count": self._expected_slide_count,
            }

    def poll_loop(self, running_flag: callable) -> None:
        interval = 1.0
        while running_flag():
            try:
                self.refresh()
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                    self._connected = False
            time.sleep(interval)

    def _request_state(self) -> dict[str, Any]:
        url = f"{self._base_url}/api/subtitle-state"
        req = Request(url, method="GET")
        try:
            with urlopen(req, timeout=self._request_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (URLError, HTTPError, OSError, json.JSONDecodeError) as exc:
            raise SubtitleSyncUnavailableError(f"Failed to reach projection server: {exc}") from exc
        if not isinstance(data, dict):
            raise SubtitleStateValidationError("Response is not a JSON object")
        return data

    def _validate_state(self, state: dict[str, Any]) -> None:
        pid = state.get("playlist_id")
        if pid != self._expected_playlist_id:
            raise SubtitlePlaylistMismatchError(
                f"playlist_id mismatch: expected {self._expected_playlist_id!r}, got {pid!r}"
            )
        count = state.get("slide_count")
        if int(count) != self._expected_slide_count:
            raise SubtitlePlaylistMismatchError(
                f"slide_count mismatch: expected {self._expected_slide_count}, got {count}"
            )
        seq = state.get("sequence_no")
        if not isinstance(seq, int) or not (1 <= seq <= self._expected_slide_count):
            raise SubtitleStateValidationError(
                f"sequence_no out of range: {seq} (valid: 1-{self._expected_slide_count})"
            )
