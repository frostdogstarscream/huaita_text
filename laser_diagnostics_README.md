# Laser Diagnostics

This project now includes a standalone diagnostic script for the SDBM-60 laser distance sensor over UART.

## Run

From `D:\Code\huaita_text`:

```powershell
python laser_diagnostics.py --port COM3 --duration 10
```

If you want to stop after a fixed number of valid frames:

```powershell
python laser_diagnostics.py --port COM3 --max-frames 20
```

## What it prints

Each valid frame prints:

- timestamp
- raw frame in hex
- parsed `distance_cm`
- `quality_high` and `quality_low`
- whether the reading is in trigger range
- stable sample count
- whether the stable trigger condition is met
- current trigger state and trigger count
- checksum expected/actual comparison

Invalid or noisy data prints an error line with the reason.

At this stage, checksum mismatch is printed as an observation and does not block distance output. This helps confirm the real frame layout before the final checksum rule is locked in.

## Current blocker observed on this machine

At the moment, opening `COM3` returns:

```text
PermissionError(13, '拒绝访问。', None, 5)
```

That means the port is being blocked at the system level, typically because:

- another program already opened `COM3`
- the device driver is not ready
- the serial adapter is in a bad state and needs reconnecting

## Next checks

1. Close any serial tools, vendor tools, or old Python services that may already hold `COM3`.
2. Replug the CH340 device and confirm it still appears as `COM3`.
3. Re-run the diagnostic command above.
