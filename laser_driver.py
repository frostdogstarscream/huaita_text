from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

try:
    import serial
    from serial.tools import list_ports
except ModuleNotFoundError:  # pragma: no cover - depends on host env
    serial = None
    list_ports = None


class LaserUnavailableError(RuntimeError):
    pass


class LaserDriver:
    START_CONTINUOUS_FAST = bytes.fromhex("AA0000200001000627")
    STOP_CONTINUOUS = b"\x58"
    RESULT_FRAME_PREFIX = bytes.fromhex("AA0000220003")
    RESULT_FRAME_LENGTH = 13
    CHECKSUM_OFFSET = 0x56
    CH340_VID = 0x1A86
    CH340_PID = 0x7523

    def __init__(self, laser_config: dict[str, Any]) -> None:
        self.enabled = bool(laser_config.get("enabled", False))
        self.configured_serial_port = str(laser_config.get("serial_port", "COM3"))
        self.serial_port = self.configured_serial_port
        self.active_serial_port = self.serial_port
        self.auto_detected = False
        self._serial_port_persist_callback: Optional[Callable[[str], None]] = None

        self.baudrate = int(laser_config.get("baudrate", 19200))
        self.bytesize = int(laser_config.get("bytesize", 8))
        self.stopbits = int(laser_config.get("stopbits", 1))
        self.parity = str(laser_config.get("parity", "N")).upper()
        self.timeout_seconds = float(laser_config.get("timeout_seconds", 0.2))
        self.measure_mode = str(laser_config.get("measure_mode", "continuous_fast_20hz"))

        self.trigger_min_cm = float(laser_config.get("trigger_min_cm", laser_config.get("trigger_min_mm", 800) / 10))
        self.trigger_max_cm = float(laser_config.get("trigger_max_cm", laser_config.get("trigger_max_mm", 1500) / 10))
        self.stable_samples = int(laser_config.get("stable_samples", 3))
        self.stable_delta_cm = float(
            laser_config.get("stable_delta_cm", laser_config.get("stable_delta_mm", 50) / 10)
        )
        self.countdown_seconds = float(laser_config.get("countdown_seconds", 5))
        self.cooldown_ms = int(laser_config.get("cooldown_ms", 5000))
        self.require_leave_before_retrigger = bool(laser_config.get("require_leave_before_retrigger", True))
        self.leave_min_cm = float(laser_config.get("leave_min_cm", laser_config.get("leave_min_mm", 1800) / 10))
        self.keepalive_enabled = bool(laser_config.get("keepalive_enabled", True))
        self.keepalive_no_data_seconds = float(laser_config.get("keepalive_no_data_seconds", 3.0))
        self.reconnect_no_data_seconds = float(laser_config.get("reconnect_no_data_seconds", 8.0))

        self.running = False
        self.connected = False
        self.worker: threading.Thread | None = None
        self.serial_conn: Any = None
        self.buffer = bytearray()

        self.distance_cm: float | None = None
        self.signal_quality_high: int | None = None
        self.signal_quality_low: int | None = None
        self.checksum_expected: int | None = None
        self.checksum_actual: int | None = None
        self.checksum_ok: bool | None = None
        self.last_frame_time = 0.0
        self.last_start_command_time = 0.0
        self.connection_started_at = 0.0
        self.keepalive_count = 0
        self.reconnect_count = 0
        self.last_error = ""
        self.trigger_count = 0
        self.pending_trigger = False

        self.trigger_state = "MANUAL_ONLY" if not self.enabled else "NOT_IMPLEMENTED"
        self.cooldown_until = 0.0
        self.awaiting_reset = False
        self.last_out_of_range = False
        self.countdown_started_at: float | None = None
        self.history: deque[float] = deque(maxlen=max(self.stable_samples, 1))
        self.state_lock = threading.RLock()

    def set_serial_port_persist_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        self._serial_port_persist_callback = callback

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        if not self.enabled:
            self.trigger_state = "MANUAL_ONLY"
            return
        if serial is None:
            self.trigger_state = "DRIVER_UNAVAILABLE"
            self.last_error = "pyserial is not installed"
            return
        self.worker = threading.Thread(target=self._reader_loop, daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.running = False
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=2)
        self._close_serial()

    def tick(self) -> None:
        with self.state_lock:
            if self.trigger_state != "COUNTDOWN" or self.countdown_started_at is None:
                return
            if self._countdown_remaining_seconds_unlocked() > 0:
                return
            self.countdown_started_at = None
            self.pending_trigger = True
            self.trigger_count += 1
            self.trigger_state = "COOLDOWN"
            self.cooldown_until = time.time() + (self.cooldown_ms / 1000.0)
            self.awaiting_reset = True
            self.last_out_of_range = False

    def can_trigger(self) -> bool:
        with self.state_lock:
            return self._can_trigger_unlocked()

    def reset_trigger_flow(self) -> dict[str, Any]:
        with self.state_lock:
            self.pending_trigger = False
            self.cooldown_until = 0.0
            self.awaiting_reset = False
            self.last_out_of_range = False
            self.history.clear()
            self._cancel_countdown_unlocked()
            if not self.enabled:
                self.trigger_state = "MANUAL_ONLY"
            elif self.connected:
                self.trigger_state = "IDLE"
            elif serial is None:
                self.trigger_state = "DRIVER_UNAVAILABLE"
            else:
                self.trigger_state = "PORT_UNAVAILABLE"
            return dict(self.status())

    def consume_trigger(self) -> bool:
        with self.state_lock:
            if self.pending_trigger:
                self.pending_trigger = False
                return True
            return False

    def status(self) -> dict[str, Any]:
        with self.state_lock:
            countdown_remaining = self._countdown_remaining_display_unlocked()
            if not self.enabled:
                return {
                    "enabled": False,
                    "connected": False,
                    "configured_serial_port": self.configured_serial_port,
                    "active_serial_port": self.active_serial_port,
                    "distance_cm": None,
                    "trigger_min_cm": self.trigger_min_cm,
                    "trigger_max_cm": self.trigger_max_cm,
                    "person_in_range": False,
                    "signal_quality_high": None,
                    "signal_quality_low": None,
                    "checksum_expected": None,
                    "checksum_actual": None,
                    "checksum_ok": None,
                    "auto_detected": False,
                    "trigger_state": "MANUAL_ONLY",
                    "countdown_active": False,
                    "countdown_seconds": self.countdown_seconds,
                    "countdown_remaining": 0,
                    "can_trigger": True,
                    "message": "当前为手动拍照模式，请使用页面按钮触发拍照。",
                    "last_error": self.last_error,
                    "last_frame_time": self.last_frame_time,
                    "no_data_seconds": None,
                    "last_start_command_time": self.last_start_command_time,
                    "keepalive_enabled": self.keepalive_enabled,
                    "keepalive_count": self.keepalive_count,
                    "reconnect_count": self.reconnect_count,
                    "measure_mode": self.measure_mode,
                    "trigger_count": self.trigger_count,
                }

            d = self.distance_cm
            person = (
                d is not None
                and math.isfinite(float(d))
                and self.trigger_min_cm <= float(d) <= self.trigger_max_cm
            )
            return {
                "enabled": True,
                "connected": self.connected,
                "configured_serial_port": self.configured_serial_port,
                "active_serial_port": self.active_serial_port,
                "distance_cm": self.distance_cm,
                "trigger_min_cm": self.trigger_min_cm,
                "trigger_max_cm": self.trigger_max_cm,
                "person_in_range": person,
                "signal_quality_high": self.signal_quality_high,
                "signal_quality_low": self.signal_quality_low,
                "checksum_expected": self.checksum_expected,
                "checksum_actual": self.checksum_actual,
                "checksum_ok": self.checksum_ok,
                "auto_detected": self.auto_detected,
                "trigger_state": self.trigger_state,
                "countdown_active": self.trigger_state == "COUNTDOWN",
                "countdown_seconds": self.countdown_seconds,
                "countdown_remaining": countdown_remaining,
                "can_trigger": self._can_trigger_unlocked(),
                "message": self._build_message_unlocked(countdown_remaining),
                "last_error": self.last_error,
                "last_frame_time": self.last_frame_time,
                "no_data_seconds": self._no_data_seconds_unlocked(),
                "last_start_command_time": self.last_start_command_time,
                "keepalive_enabled": self.keepalive_enabled,
                "keepalive_count": self.keepalive_count,
                "reconnect_count": self.reconnect_count,
                "measure_mode": self.measure_mode,
                "trigger_count": self.trigger_count,
            }

    def _can_trigger_unlocked(self) -> bool:
        if not self.enabled:
            return True
        if not self.connected:
            return False
        return self.trigger_state in {"IDLE", "COUNTDOWN"}

    def _countdown_remaining_seconds_unlocked(self) -> float:
        if self.countdown_started_at is None:
            return 0.0
        elapsed = time.time() - self.countdown_started_at
        return max(self.countdown_seconds - elapsed, 0.0)

    def _countdown_remaining_display_unlocked(self) -> int:
        remaining = self._countdown_remaining_seconds_unlocked()
        return int(math.ceil(remaining)) if remaining > 0 else 0

    def _no_data_seconds_unlocked(self) -> float | None:
        start_time = self.last_frame_time or self.connection_started_at
        if not start_time:
            return None
        return max(time.time() - start_time, 0.0)

    def _cancel_countdown_unlocked(self) -> None:
        self.countdown_started_at = None

    def _build_message_unlocked(self, countdown_remaining: int) -> str:
        if serial is None:
            return "缺少 pyserial，无法启用真实激光串口测距。"
        if self.trigger_state == "PORT_UNAVAILABLE":
            if self.last_error == "未发现可用的 CH340 激光串口":
                return self.last_error
            if self.auto_detected and self.active_serial_port:
                return f"已自动切换到 {self.active_serial_port}，激光测距运行中。"
            return f"串口 {self.configured_serial_port} 打开失败：{self.last_error or 'unknown error'}"
        if self.trigger_state == "DRIVER_UNAVAILABLE":
            return "缺少 pyserial，无法启用真实激光串口测距。"
        if self.trigger_state == "NOT_IMPLEMENTED":
            return "激光驱动已启用，但测量模式未实现。"
        if self.trigger_state == "READ_ERROR":
            return f"串口读取异常：{self.last_error or 'unknown error'}"
        if self.trigger_state == "COUNTDOWN":
            return f"人物已进入标准位置，{countdown_remaining} 秒后立即拍照。"
        if self.trigger_state == "COOLDOWN":
            if self.awaiting_reset:
                if self.require_leave_before_retrigger:
                    return f"拍照完成，请离开到 {self.leave_min_cm:.0f}cm 以外后进入下一轮。"
                return "拍照完成，冷却结束后将自动进入下一轮。"
            return "激光触发已命中，冷却中。"
        if self.trigger_state == "TRIGGERED":
            return "已产生一次激光触发事件，等待应用层消费。"
        if self.trigger_state == "IDLE":
            if self.last_out_of_range:
                return "目标已离开触发范围，等待重新进入。"
            if self.connected and self.distance_cm is None:
                if self.auto_detected and self.active_serial_port:
                    return f"已自动切换到 {self.active_serial_port}，激光测距已连接，正在等待首帧数据。"
                return "激光测距已连接，正在等待首帧数据。"
            if self.auto_detected and self.active_serial_port:
                return f"已自动切换到 {self.active_serial_port}，激光测距运行中，等待目标进入触发区。"
            return "激光测距运行中，等待目标进入触发区。"
        return self.last_error or "激光驱动运行中。"

    def _is_ch340_port(self, port_info: Any) -> bool:
        description = str(getattr(port_info, "description", "") or "").upper()
        manufacturer = str(getattr(port_info, "manufacturer", "") or "").upper()
        hwid = str(getattr(port_info, "hwid", "") or "").upper()
        vid = getattr(port_info, "vid", None)
        pid = getattr(port_info, "pid", None)
        return (
            "CH340" in description
            or "CH340" in manufacturer
            or "1A86:7523" in hwid
            or (vid == self.CH340_VID and pid == self.CH340_PID)
        )

    def _discover_candidate_ports(self) -> list[str]:
        candidates: list[str] = []
        configured = self.configured_serial_port.strip()
        if configured:
            candidates.append(configured)

        if list_ports is None:
            return candidates

        seen = {configured} if configured else set()
        for port_info in list_ports.comports():
            device = str(getattr(port_info, "device", "") or "")
            if not device or device in seen:
                continue
            if self._is_ch340_port(port_info):
                candidates.append(device)
                seen.add(device)
        return candidates

    def _probe_connection(self, conn: Any) -> bool:
        probe_buffer = bytearray()
        deadline = time.time() + max(self.timeout_seconds * 8, 1.0)
        while time.time() < deadline:
            chunk = conn.read(64)
            if not chunk:
                continue
            probe_buffer.extend(chunk)
            while len(probe_buffer) >= self.RESULT_FRAME_LENGTH:
                prefix_index = probe_buffer.find(self.RESULT_FRAME_PREFIX)
                if prefix_index < 0:
                    probe_buffer[:] = probe_buffer[-(len(self.RESULT_FRAME_PREFIX) - 1) :]
                    break
                if prefix_index > 0:
                    del probe_buffer[:prefix_index]
                if len(probe_buffer) < self.RESULT_FRAME_LENGTH:
                    break
                frame = bytes(probe_buffer[: self.RESULT_FRAME_LENGTH])
                del probe_buffer[: self.RESULT_FRAME_LENGTH]
                self.parse_measurement_frame(frame)
                return True
        return False

    def _send_start_continuous(self, conn: Any) -> None:
        conn.write(self.START_CONTINUOUS_FAST)
        with self.state_lock:
            self.last_start_command_time = time.time()

    def _open_serial(self) -> None:
        if serial is None:
            raise LaserUnavailableError("pyserial is not installed")
        self._close_serial()

        candidates = self._discover_candidate_ports()
        if not candidates:
            raise LaserUnavailableError("未发现可用的 CH340 激光串口")

        parity = {"N": serial.PARITY_NONE, "E": serial.PARITY_EVEN, "O": serial.PARITY_ODD}.get(
            self.parity, serial.PARITY_NONE
        )
        configured = self.configured_serial_port.strip()
        last_error: Exception | None = None

        for port_name in candidates:
            conn = None
            try:
                conn = serial.Serial(
                    port=port_name,
                    baudrate=self.baudrate,
                    bytesize=self.bytesize,
                    parity=parity,
                    stopbits=self.stopbits,
                    timeout=self.timeout_seconds,
                )
                conn.reset_input_buffer()
                conn.reset_output_buffer()
                if self.measure_mode != "continuous_fast_20hz":
                    raise LaserUnavailableError(f"Unsupported measure mode: {self.measure_mode}")
                with self.state_lock:
                    self.connection_started_at = time.time()
                    self.last_frame_time = 0.0
                self._send_start_continuous(conn)

                # 首帧只作为探测增强，不作为连接成功的硬性条件。
                self._probe_connection(conn)

                with self.state_lock:
                    self.serial_conn = conn
                    self.connected = True
                    self.serial_port = port_name
                    self.active_serial_port = port_name
                    self.auto_detected = port_name != configured
                    self.trigger_state = "IDLE"
                    self.last_error = "等待首帧数据中"
                    self.buffer.clear()
                    self._cancel_countdown_unlocked()

                if self.auto_detected and self._serial_port_persist_callback is not None:
                    try:
                        self._serial_port_persist_callback(port_name)
                        with self.state_lock:
                            self.configured_serial_port = port_name
                    except Exception as exc:
                        with self.state_lock:
                            self.last_error = str(exc)
                return
            except Exception as exc:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                last_error = exc

        if last_error is None:
            raise LaserUnavailableError("未发现可用的 CH340 激光串口")
        raise LaserUnavailableError(str(last_error))

    def _reader_loop(self) -> None:
        while self.running:
            if not self.connected:
                try:
                    self._open_serial()
                except Exception as exc:  # pragma: no cover - depends on hardware
                    with self.state_lock:
                        self.connected = False
                        self.trigger_state = "PORT_UNAVAILABLE"
                        self.last_error = str(exc)
                        self._cancel_countdown_unlocked()
                    time.sleep(1.0)
                    continue

            try:
                chunk = self.serial_conn.read(64)
                if chunk:
                    self.buffer.extend(chunk)
                    self._drain_buffer()
                else:
                    self._handle_no_data()
            except Exception as exc:  # pragma: no cover - depends on hardware
                with self.state_lock:
                    self.last_error = str(exc)
                    self.trigger_state = "READ_ERROR"
                    self._cancel_countdown_unlocked()
                self._close_serial()
                time.sleep(0.5)

    def _close_serial(self) -> None:
        conn = self.serial_conn
        self.serial_conn = None
        if conn is not None:
            try:
                conn.write(self.STOP_CONTINUOUS)
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        with self.state_lock:
            self.connected = False

    def _handle_no_data(self) -> None:
        should_reconnect = False
        should_keepalive = False
        with self.state_lock:
            if (
                self.trigger_state == "COOLDOWN"
                and not self.awaiting_reset
                and time.time() >= self.cooldown_until
            ):
                self.trigger_state = "IDLE"
            no_data_seconds = self._no_data_seconds_unlocked()
            if no_data_seconds is None:
                return
            if (
                self.reconnect_no_data_seconds > 0
                and no_data_seconds >= self.reconnect_no_data_seconds
            ):
                self.reconnect_count += 1
                self.last_error = f"no laser frames for {no_data_seconds:.1f}s; reconnecting"
                self._cancel_countdown_unlocked()
                should_reconnect = True
            elif (
                self.keepalive_enabled
                and self.keepalive_no_data_seconds > 0
                and no_data_seconds >= self.keepalive_no_data_seconds
                and (
                    not self.last_start_command_time
                    or time.time() - self.last_start_command_time >= self.keepalive_no_data_seconds
                )
            ):
                should_keepalive = True

        if should_reconnect:
            self._close_serial()
            return

        if should_keepalive and self.serial_conn is not None:
            try:
                self._send_start_continuous(self.serial_conn)
                with self.state_lock:
                    self.keepalive_count += 1
                    self.last_error = f"resent continuous measurement after {no_data_seconds:.1f}s without frames"
            except Exception as exc:
                with self.state_lock:
                    self.last_error = f"keepalive command failed: {exc}"

    def _drain_buffer(self) -> None:
        while len(self.buffer) >= self.RESULT_FRAME_LENGTH:
            prefix_index = self.buffer.find(self.RESULT_FRAME_PREFIX)
            if prefix_index < 0:
                self.buffer[:] = self.buffer[-(len(self.RESULT_FRAME_PREFIX) - 1) :]
                return
            if prefix_index > 0:
                del self.buffer[:prefix_index]
            if len(self.buffer) < self.RESULT_FRAME_LENGTH:
                return
            frame = bytes(self.buffer[: self.RESULT_FRAME_LENGTH])
            del self.buffer[: self.RESULT_FRAME_LENGTH]
            try:
                parsed = self.parse_measurement_frame(frame)
            except ValueError as exc:
                with self.state_lock:
                    self.last_error = str(exc)
                continue
            with self.state_lock:
                self.signal_quality_high = parsed["signal_quality_high"]
                self.signal_quality_low = parsed["signal_quality_low"]
                self.checksum_expected = parsed["checksum_expected"]
                self.checksum_actual = parsed["checksum_actual"]
                self.checksum_ok = parsed["checksum_ok"]
            self._handle_measurement(float(parsed["distance_cm"]))

    def parse_measurement_frame(self, frame: bytes) -> dict[str, Any]:
        if len(frame) != self.RESULT_FRAME_LENGTH:
            raise ValueError(f"Unexpected frame length: {len(frame)}")
        if frame[:6] != self.RESULT_FRAME_PREFIX:
            raise ValueError("Invalid frame prefix")
        distance_raw_mm = int.from_bytes(frame[6:10], byteorder="big", signed=False)
        if distance_raw_mm == 0:
            raise ValueError("Distance value is zero")
        checksum_expected = (sum(frame[:12]) + self.CHECKSUM_OFFSET) & 0xFF
        checksum_actual = frame[12]
        return {
            "distance_cm": round(distance_raw_mm / 10.0, 1),
            "distance_raw_mm": distance_raw_mm,
            "signal_quality_high": frame[10],
            "signal_quality_low": frame[11],
            "checksum_expected": checksum_expected,
            "checksum_actual": checksum_actual,
            "checksum_ok": checksum_expected == checksum_actual,
        }

    def _parse_measurement_frame(self, frame: bytes) -> int:
        return int(self.parse_measurement_frame(frame)["distance_raw_mm"])

    def _handle_measurement(self, distance_cm: float) -> None:
        now = time.time()
        with self.state_lock:
            self.distance_cm = distance_cm
            self.last_frame_time = now

            if self.trigger_state == "COOLDOWN":
                if self.awaiting_reset:
                    if self.require_leave_before_retrigger:
                        if distance_cm >= self.leave_min_cm:
                            self.awaiting_reset = False
                            self.last_out_of_range = True
                        else:
                            return
                    elif now >= self.cooldown_until:
                        self.awaiting_reset = False
                    else:
                        return
                if self.awaiting_reset:
                    return
                if now < self.cooldown_until:
                    return
                self.trigger_state = "IDLE"

            in_range = self.trigger_min_cm <= distance_cm <= self.trigger_max_cm
            if not in_range:
                self.history.clear()
                if self.trigger_state == "COUNTDOWN":
                    self.last_out_of_range = True
                self._cancel_countdown_unlocked()
                self.trigger_state = "IDLE"
                return

            if self.trigger_state == "COUNTDOWN":
                return

            self.last_out_of_range = False

            if self.countdown_seconds <= 0:
                self.pending_trigger = True
                self.trigger_count += 1
                self.trigger_state = "COOLDOWN"
                self.cooldown_until = now + (self.cooldown_ms / 1000.0)
                self.awaiting_reset = True
                return

            self.countdown_started_at = now
            self.trigger_state = "COUNTDOWN"
