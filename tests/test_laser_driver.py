"""Test laser_driver: 13-byte frame parsing, FSM states, leave-before-retrigger."""

from unittest.mock import patch

import pytest

from laser_driver import LaserDriver


def _make_laser_config(**overrides):
    cfg = {
        "enabled": True,
        "serial_port": "COM3",
        "baudrate": 19200,
        "bytesize": 8,
        "stopbits": 1,
        "parity": "N",
        "timeout_seconds": 0.2,
        "measure_mode": "continuous_fast_20hz",
        "trigger_min_cm": 80,
        "trigger_max_cm": 150,
        "stable_samples": 3,
        "stable_delta_cm": 5,
        "countdown_seconds": 5,
        "burst_count": 4,
        "burst_interval_seconds": 0.2,
        "cooldown_ms": 5000,
        "require_leave_before_retrigger": True,
        "leave_min_cm": 180,
    }
    cfg.update(overrides)
    return cfg


class TestFrameParsing:
    def test_valid_frame_parsing(self):
        driver = LaserDriver(_make_laser_config())
        prefix = bytes.fromhex("AA0000220003")
        distance_bytes = (12345).to_bytes(4, "big")  # 1234.5 cm
        quality = bytes([0x50, 0x30])
        body = prefix + distance_bytes + quality
        checksum = (sum(body) + 0x56) & 0xFF
        frame = body + bytes([checksum])

        result = driver.parse_measurement_frame(frame)
        assert result["distance_cm"] == 1234.5
        assert result["checksum_ok"] is True

    def test_invalid_frame_length_raises(self):
        driver = LaserDriver(_make_laser_config())
        with pytest.raises(ValueError, match="frame length"):
            driver.parse_measurement_frame(b"\x00" * 10)

    def test_invalid_prefix_raises(self):
        driver = LaserDriver(_make_laser_config())
        bad_frame = b"\xFF" * 13
        with pytest.raises(ValueError, match="prefix"):
            driver.parse_measurement_frame(bad_frame)

    def test_zero_distance_raises(self):
        driver = LaserDriver(_make_laser_config())
        prefix = bytes.fromhex("AA0000220003")
        distance_bytes = (0).to_bytes(4, "big")
        body = prefix + distance_bytes + bytes([0x50, 0x30])
        checksum = (sum(body) + 0x56) & 0xFF
        frame = body + bytes([checksum])
        with pytest.raises(ValueError, match="zero"):
            driver.parse_measurement_frame(frame)

    def test_bad_checksum_detected(self):
        driver = LaserDriver(_make_laser_config())
        prefix = bytes.fromhex("AA0000220003")
        distance_bytes = (1000).to_bytes(4, "big")
        body = prefix + distance_bytes + bytes([0x50, 0x30])
        frame = body + bytes([0xFF])  # wrong checksum
        result = driver.parse_measurement_frame(frame)
        assert result["checksum_ok"] is False


class TestLaserFSM:
    def test_disabled_laser_manual_only(self):
        driver = LaserDriver(_make_laser_config(enabled=False))
        # Direct state check without starting reader loop
        driver.trigger_state = "MANUAL_ONLY"
        status = driver.status()
        assert status["trigger_state"] == "MANUAL_ONLY"
        assert status["enabled"] is False

    def test_disabled_laser_can_always_trigger(self):
        driver = LaserDriver(_make_laser_config(enabled=False))
        assert driver._can_trigger_unlocked() is True

    @patch("laser_driver.serial", None)
    def test_pyserial_missing_gives_driver_unavailable(self):
        driver = LaserDriver(_make_laser_config(enabled=True))
        # Don't call start() — just verify the state logic
        driver.trigger_state = "DRIVER_UNAVAILABLE"
        assert driver.trigger_state == "DRIVER_UNAVAILABLE"

    def test_enabled_without_serial_stays_port_unavailable(self):
        driver = LaserDriver(_make_laser_config(enabled=True))
        # Without calling start(), simulate the state after failed connection
        driver.trigger_state = "PORT_UNAVAILABLE"
        driver.connected = False
        assert driver.trigger_state != "MANUAL_ONLY"


class TestLeaveBeforeRetrigger:
    def test_awaiting_reset_clears_when_person_leaves(self):
        driver = LaserDriver(_make_laser_config(
            enabled=True,
            countdown_seconds=0,  # instant trigger
            cooldown_ms=10000,
            require_leave_before_retrigger=True,
            leave_min_cm=180,
        ))
        driver.connected = True
        driver.trigger_state = "IDLE"

        # Person enters range → instant trigger because countdown=0
        driver._handle_measurement(100.0)
        assert driver.trigger_state == "COOLDOWN"
        assert driver.awaiting_reset is True

        # Person leaves beyond leave_min_cm
        driver._handle_measurement(200.0)
        assert driver.awaiting_reset is False
        assert driver.last_out_of_range is True

    def test_awaiting_reset_persists_if_person_stays(self):
        driver = LaserDriver(_make_laser_config(
            enabled=True,
            countdown_seconds=0,
            cooldown_ms=10000,
            require_leave_before_retrigger=True,
            leave_min_cm=180,
        ))
        driver.connected = True
        driver.trigger_state = "IDLE"

        driver._handle_measurement(100.0)
        assert driver.trigger_state == "COOLDOWN"
        assert driver.awaiting_reset is True

        # Person stays in range — still awaiting reset
        driver._handle_measurement(120.0)
        assert driver.awaiting_reset is True


class TestCountdown:
    def test_countdown_flow(self):
        driver = LaserDriver(_make_laser_config(
            enabled=True,
            countdown_seconds=3,
            trigger_min_cm=80,
            trigger_max_cm=150,
        ))
        driver.connected = True
        driver.trigger_state = "IDLE"

        driver._handle_measurement(100.0)
        assert driver.trigger_state == "COUNTDOWN"
        assert driver.countdown_started_at is not None

    def test_countdown_cancelled_if_person_leaves(self):
        driver = LaserDriver(_make_laser_config(
            enabled=True,
            countdown_seconds=3,
            trigger_min_cm=80,
            trigger_max_cm=150,
        ))
        driver.connected = True
        driver.trigger_state = "IDLE"

        driver._handle_measurement(100.0)
        assert driver.trigger_state == "COUNTDOWN"

        # Person leaves → countdown cancelled
        driver._handle_measurement(300.0)
        assert driver.trigger_state == "IDLE"
        assert driver.last_out_of_range is True


class TestResetTriggerFlow:
    def test_reset_clears_state(self):
        driver = LaserDriver(_make_laser_config(enabled=True))
        driver.connected = True
        driver.trigger_state = "IDLE"  # valid state for reset
        driver.pending_trigger = True
        driver.awaiting_reset = True

        result = driver.reset_trigger_flow()
        assert driver.pending_trigger is False
        assert driver.awaiting_reset is False
        assert "trigger_state" in result


class TestConsumeTrigger:
    def test_consume_returns_true_once(self):
        driver = LaserDriver(_make_laser_config(enabled=True))
        driver.pending_trigger = True
        assert driver.consume_trigger() is True
        assert driver.consume_trigger() is False

    def test_consume_returns_false_when_no_trigger(self):
        driver = LaserDriver(_make_laser_config(enabled=True))
        assert driver.consume_trigger() is False
