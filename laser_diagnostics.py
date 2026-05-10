from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import serial

from laser_driver import LaserDriver


def load_laser_config(config_path: Path) -> dict[str, Any]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    laser_config = raw.get("laser_trigger")
    if not isinstance(laser_config, dict):
        raise ValueError(f"Missing laser_trigger config in {config_path}")
    return laser_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnostic reader for SDBM-60 laser distance data over UART."
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config.json. Default: %(default)s",
    )
    parser.add_argument(
        "--port",
        default=None,
        help="Override serial port. Default comes from config.json",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override serial read timeout in seconds. Default comes from config.json",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many valid measurement frames. 0 means run until interrupted.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop after this many seconds. 0 means run until interrupted.",
    )
    return parser


def format_frame(frame: bytes) -> str:
    return " ".join(f"{byte:02X}" for byte in frame)


def format_timestamp(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) + f".{int((ts % 1) * 1000):03d}"


def format_status(driver: LaserDriver) -> str:
    in_range = (
        driver.distance_cm is not None
        and driver.trigger_min_cm <= driver.distance_cm <= driver.trigger_max_cm
    )
    stable_count = len(driver.history)
    stable_ready = (
        stable_count >= driver.stable_samples
        and max(driver.history) - min(driver.history) <= driver.stable_delta_cm
    )
    return (
        f"in_range={in_range} "
        f"stable_count={stable_count}/{driver.stable_samples} "
        f"stable_ready={stable_ready} "
        f"trigger_state={driver.trigger_state} "
        f"trigger_count={driver.trigger_count}"
    )


def format_checksum(parsed: dict[str, Any]) -> str:
    return (
        f"checksum_ok={parsed['checksum_ok']} "
        f"checksum_expected=0x{parsed['checksum_expected']:02X} "
        f"checksum_actual=0x{parsed['checksum_actual']:02X}"
    )


def print_config_summary(driver: LaserDriver) -> None:
    print("laser diagnostic started")
    print(f"port={driver.serial_port}")
    print(
        "serial="
        f"{driver.baudrate}bps "
        f"bytesize={driver.bytesize} "
        f"stopbits={driver.stopbits} "
        f"parity={driver.parity} "
        f"timeout={driver.timeout_seconds}s"
    )
    print(
        "trigger="
        f"{driver.trigger_min_cm}-{driver.trigger_max_cm}cm "
        f"stable_samples={driver.stable_samples} "
        f"stable_delta={driver.stable_delta_cm}cm "
        f"cooldown={driver.cooldown_ms}ms "
        f"leave_min={driver.leave_min_cm}cm"
    )
    print(f"start_cmd={format_frame(driver.START_CONTINUOUS_FAST)} stop_cmd={format_frame(driver.STOP_CONTINUOUS)}")
    print("waiting for frames...")


def open_serial(driver: LaserDriver) -> serial.Serial:
    parity_map = {
        "N": serial.PARITY_NONE,
        "E": serial.PARITY_EVEN,
        "O": serial.PARITY_ODD,
    }
    conn = serial.Serial(
        port=driver.serial_port,
        baudrate=driver.baudrate,
        bytesize=driver.bytesize,
        parity=parity_map.get(driver.parity, serial.PARITY_NONE),
        stopbits=driver.stopbits,
        timeout=driver.timeout_seconds,
    )
    conn.reset_input_buffer()
    conn.reset_output_buffer()
    conn.write(driver.START_CONTINUOUS_FAST)
    conn.flush()
    return conn


def iter_measurement_frames(buffer: bytearray, frame_length: int, prefix: bytes) -> tuple[list[bytes], list[str]]:
    frames: list[bytes] = []
    errors: list[str] = []
    while len(buffer) >= frame_length:
        prefix_index = buffer.find(prefix)
        if prefix_index < 0:
            if len(buffer) > len(prefix) - 1:
                dropped = bytes(buffer[: len(buffer) - (len(prefix) - 1)])
                del buffer[: len(buffer) - (len(prefix) - 1)]
                errors.append(f"dropped_noise={format_frame(dropped)}")
            return frames, errors
        if prefix_index > 0:
            noise = bytes(buffer[:prefix_index])
            del buffer[:prefix_index]
            errors.append(f"dropped_noise={format_frame(noise)}")
        if len(buffer) < frame_length:
            return frames, errors
        frames.append(bytes(buffer[:frame_length]))
        del buffer[:frame_length]
    return frames, errors


def run_diagnostics(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    laser_config = load_laser_config(config_path)
    if args.port:
        laser_config["serial_port"] = args.port
    if args.timeout is not None:
        laser_config["timeout_seconds"] = args.timeout

    driver = LaserDriver(laser_config)
    if driver.measure_mode != "continuous_fast_20hz":
        raise ValueError(f"Unsupported measure mode: {driver.measure_mode}")

    print_config_summary(driver)

    buffer = bytearray()
    valid_frames = 0
    start_time = time.time()
    conn: serial.Serial | None = None

    try:
        conn = open_serial(driver)
        print(f"serial connected: {driver.serial_port}")

        while True:
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                print("duration reached, stopping diagnostics")
                break

            chunk = conn.read(64)
            if not chunk:
                now = format_timestamp(time.time())
                print(f"{now} no_data timeout={driver.timeout_seconds}s")
                continue

            buffer.extend(chunk)
            frames, errors = iter_measurement_frames(
                buffer,
                driver.RESULT_FRAME_LENGTH,
                driver.RESULT_FRAME_PREFIX,
            )

            for message in errors:
                print(f"{format_timestamp(time.time())} {message}")

            for frame in frames:
                timestamp = time.time()
                frame_hex = format_frame(frame)
                try:
                    parsed = driver.parse_measurement_frame(frame)
                except ValueError as exc:
                    print(f"{format_timestamp(timestamp)} frame={frame_hex} valid=false error={exc}")
                    continue

                driver.signal_quality_high = parsed["signal_quality_high"]
                driver.signal_quality_low = parsed["signal_quality_low"]
                driver.checksum_expected = parsed["checksum_expected"]
                driver.checksum_actual = parsed["checksum_actual"]
                driver.checksum_ok = parsed["checksum_ok"]
                driver._handle_measurement(float(parsed["distance_cm"]))
                valid_frames += 1
                print(
                    f"{format_timestamp(timestamp)} "
                    f"frame={frame_hex} "
                    f"valid=true "
                    f"distance_cm={parsed['distance_cm']} "
                    f"quality_high=0x{parsed['signal_quality_high']:02X} "
                    f"quality_low=0x{parsed['signal_quality_low']:02X} "
                    f"{format_checksum(parsed)} "
                    f"{format_status(driver)}"
                )

                if args.max_frames > 0 and valid_frames >= args.max_frames:
                    print("max_frames reached, stopping diagnostics")
                    return 0

    except serial.SerialException as exc:
        print(f"serial open/read failed on {driver.serial_port}: {exc}")
        return 1
    finally:
        if conn is not None:
            try:
                conn.write(driver.STOP_CONTINUOUS)
                conn.flush()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        print("serial closed")

    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_diagnostics(args)


if __name__ == "__main__":
    raise SystemExit(main())
