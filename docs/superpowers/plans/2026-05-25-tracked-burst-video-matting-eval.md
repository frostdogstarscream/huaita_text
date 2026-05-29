# Tracked Burst Video Matting Offline Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline-only workflow that captures real 16-frame bursts, tracks one subject with YOLO11-seg + ByteTrack, evaluates MatAnyone2 video matting against tracked YOLO masks and 4-frame MODNet, and still outputs four composed poster images.

**Architecture:** Keep the online `capture_manager.process_capture_task()` path unchanged. Add a standalone burst collector, a sequence tracking module that converts track results into the existing instance/trimap contract, and a standalone evaluation runner that reuses current matting, edge refine, composition, and sheet-generation helpers.

**Tech Stack:** Python, OpenCV, Pillow, NumPy, Ultralytics YOLO11-seg/ByteTrack, MatAnyone2, MODNet, pytest.

---

## Working Tree Constraint

This feature must be implemented in the current experimental branch/workspace because `video_recorder.py`, `matanyone_service.py`, `modnet_matting_service.py`, and portions of the existing offline runner are present as local experimental work and are not all committed in `HEAD`. Do not reset, revert, or broadly stage the existing dirty tree. Stage only the files listed in each task; generated images, videos, weights, and local model repositories remain untracked.

## File Map

- Create `capture_burst_eval.py`: offline camera capture CLI and burst manifest writer only.
- Create `subject_instance_tracking.py`: YOLO-seg tracking adapter, subject id continuity/recovery, and tracked trimap sequence types.
- Create `run_tracked_video_matting_eval.py`: three-branch offline evaluator, metrics, sheets, and command-line entry point.
- Create `tests/test_capture_burst_eval.py`: capture manifest/output tests with a fake camera.
- Create `tests/test_subject_instance_tracking.py`: tracking continuity, recovery, failure, and visitor constraint tests with a fake backend.
- Create `tests/test_run_tracked_video_matting_eval.py`: end-to-end fake-backend runner and metric tests.
- Reuse without online behavioral changes: `video_recorder.py`, `subject_instance_segmentation.py`, `matanyone_service.py`, `modnet_matting_service.py`, `subject_edge_refine.py`, and composition helpers in `run_yolo_seg_matting_eval.py`.

### Task 1: Offline Burst Capture CLI

**Files:**
- Create: `capture_burst_eval.py`
- Create: `tests/test_capture_burst_eval.py`

- [ ] **Step 1: Write tests for a 16-frame offline take and failure on insufficient frames**

```python
from pathlib import Path
import numpy as np
import pytest

from capture_burst_eval import BurstCaptureConfig, capture_take


class FakeCamera:
    def __init__(self, available: int) -> None:
        self.available = available
        self.calls = 0

    def get_frame(self) -> np.ndarray:
        if self.calls >= self.available:
            raise RuntimeError("no frame")
        self.calls += 1
        return np.full((72, 128, 3), self.calls, dtype=np.uint8)


def test_capture_take_writes_frames_video_and_manifest(tmp_path: Path):
    cfg = BurstCaptureConfig(frame_count=16, fps=16, min_valid_frames=12, output_frame_indices=(3, 7, 10, 13))
    manifest = capture_take(FakeCamera(16), tmp_path, "visitor_crossing", "take_01", cfg, sleep_fn=lambda _: None)

    assert manifest.status == "ok"
    assert len(manifest.frame_paths) == 16
    assert manifest.output_frame_indices == [3, 7, 10, 13]
    assert (manifest.take_dir / "burst.avi").exists()
    assert (manifest.take_dir / "metadata.json").exists()


def test_capture_take_marks_take_failed_when_valid_frames_below_minimum(tmp_path: Path):
    cfg = BurstCaptureConfig(frame_count=16, fps=16, min_valid_frames=12)
    manifest = capture_take(FakeCamera(8), tmp_path, "single_subject", "take_01", cfg, sleep_fn=lambda _: None)

    assert manifest.status == "failed"
    assert manifest.valid_frame_count == 8
```

- [ ] **Step 2: Run tests and observe the missing-module failure**

Run: `python -m pytest tests/test_capture_burst_eval.py -q --basetemp=.pytest_tmp/tracked_burst_capture_red`

Expected: FAIL because `capture_burst_eval` does not exist.

- [ ] **Step 3: Implement the capture API and CLI**

Implement these public types and behavior:

```python
@dataclass(frozen=True)
class BurstCaptureConfig:
    frame_count: int = 16
    fps: int = 16
    min_valid_frames: int = 12
    resolution: tuple[int, int] = (1280, 720)
    codec: str = "mp4v"
    output_frame_indices: tuple[int, int, int, int] = (3, 7, 10, 13)


@dataclass(frozen=True)
class BurstCaptureManifest:
    take_dir: Path
    scenario: str
    take: str
    status: str
    frame_paths: list[Path]
    valid_frame_count: int
    output_frame_indices: list[int]
    video_path: Path | None
    error: str | None = None


def capture_take(camera, output_root: Path, scenario: str, take: str,
                 config: BurstCaptureConfig, *, sleep_fn=time.sleep) -> BurstCaptureManifest:
    # Save frames as frames/000001.jpg ...; feed successful frames into VideoRecorder.
    # Catch per-frame read errors, stop collecting, and still write metadata.json.
    # status="ok" only when valid_frame_count >= min_valid_frames.
```

CLI:

```powershell
python capture_burst_eval.py --scenario visitor_close_contact --take take_01 --frame-count 16 --fps 16
```

`main()` must load camera settings using `load_config()`, instantiate/start/stop `CameraDriver`, and write only beneath `generated/burst_eval_inputs/<session>/<scenario>/<take>/`. It must not import or call `capture_manager`.

- [ ] **Step 4: Run capture unit tests**

Run: `python -m pytest tests/test_capture_burst_eval.py tests/test_video_recorder.py -q --basetemp=.pytest_tmp/tracked_burst_capture_green`

Expected: PASS.

- [ ] **Step 5: Commit only the capture tool and test**

```powershell
git add -- capture_burst_eval.py tests/test_capture_burst_eval.py
git commit -m "feat: add offline burst capture utility"
```

### Task 2: YOLO-seg Tracking Sequence Adapter

**Files:**
- Create: `subject_instance_tracking.py`
- Create: `tests/test_subject_instance_tracking.py`
- Reuse: `subject_instance_segmentation.py`

- [ ] **Step 1: Write tracking behavior tests**

Use a `FakeTrackBackend` that returns per-frame `TrackedCandidate` lists and does not load Ultralytics:

```python
def test_tracking_keeps_initial_subject_id_and_marks_other_people_as_visitors(tmp_path):
    backend = FakeTrackBackend([
        [candidate(7, subject=True), candidate(9, subject=False)],
        [candidate(7, subject=True), candidate(9, subject=False)],
    ])
    sequence = SubjectInstanceTracker(config(), backend=backend).track_paths(frame_paths(tmp_path, 2))

    assert sequence.status == "ok"
    assert sequence.subject_track_id == 7
    assert [frame.selected.track_id for frame in sequence.frames] == [7, 7]
    assert all([visitor.track_id for visitor in frame.visitors] == [9] for frame in sequence.frames)


def test_tracking_recovers_subject_for_two_missing_id_frames_by_iou(tmp_path):
    backend = FakeTrackBackend([
        [candidate(7, subject=True)],
        [candidate(None, subject=True, offset=1)],
        [candidate(7, subject=True, offset=2)],
    ])
    sequence = SubjectInstanceTracker(config(max_recovery_frames=2, recovery_min_iou=0.35), backend=backend).track_paths(frame_paths(tmp_path, 3))

    assert sequence.status == "ok"
    assert sequence.frames[1].track_recovered is True
    assert sequence.track_lost_frames == 1


def test_tracking_fails_after_recovery_window_is_exceeded(tmp_path):
    backend = FakeTrackBackend([
        [candidate(7, subject=True)],
        [],
        [],
        [],
    ])
    sequence = SubjectInstanceTracker(config(max_recovery_frames=2), backend=backend).track_paths(frame_paths(tmp_path, 4))

    assert sequence.status == "tracking_failed"
```

Include an assertion that each successful `TrackedFrameResult` has populated `sure_foreground`, `sure_background`, and `unknown`, and that visitor pixels are included in `sure_background`.

- [ ] **Step 2: Run tracking tests and observe failure**

Run: `python -m pytest tests/test_subject_instance_tracking.py -q --basetemp=.pytest_tmp/subject_tracking_red`

Expected: FAIL because `subject_instance_tracking` does not exist.

- [ ] **Step 3: Implement sequence tracking types and backend**

Define:

```python
@dataclass(frozen=True)
class TrackedCandidate:
    track_id: int | None
    bbox: tuple[float, float, float, float]
    confidence: float
    mask: np.ndarray
    score: float | None = None

@dataclass(frozen=True)
class TrackedFrameResult:
    frame_path: Path
    selected: TrackedCandidate | None
    visitors: list[TrackedCandidate]
    trimap: np.ndarray
    sure_foreground: np.ndarray
    sure_background: np.ndarray
    unknown: np.ndarray
    track_recovered: bool = False

@dataclass(frozen=True)
class TrackedInstanceSequence:
    subject_track_id: int | None
    frames: list[TrackedFrameResult]
    track_switch_count: int
    track_lost_frames: int
    status: str
    error: str | None = None
```

Implement:

```python
class YoloSegTrackBackend:
    def __init__(self, model_path: str, mask_threshold: float = 0.5): ...
    def track(self, frame_path: Path, image_size: tuple[int, int]) -> list[TrackedCandidate]:
        result = self._model.track(
            source=str(frame_path),
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            save=False,
            show=False,
            verbose=False,
        )[0]
        # Resize instance masks and read result.boxes.id.


class SubjectInstanceTracker:
    def track_paths(self, frame_paths: list[Path]) -> TrackedInstanceSequence:
        # First frame: convert candidates to InstanceCandidate and reuse choose_primary_instance().
        # Following frames: prefer subject_track_id; when absent, recover using IoU with last selected.
        # Convert tracked candidates to InstanceCandidate and reuse build_instance_trimap().
```

Use `InstanceSegmentationConfig` fields for trimap radii and subject scoring. The tracker must not hard-code a direction, hair region, or visitor appearance.

- [ ] **Step 4: Run tracking and existing instance tests**

Run: `python -m pytest tests/test_subject_instance_tracking.py tests/test_subject_instance_segmentation.py -q --basetemp=.pytest_tmp/subject_tracking_green`

Expected: PASS.

- [ ] **Step 5: Commit tracking adapter**

```powershell
git add -- subject_instance_tracking.py tests/test_subject_instance_tracking.py
git commit -m "feat: add offline subject instance tracking"
```

### Task 3: Tracked Constraints And Unsupervised Metrics

**Files:**
- Create: `run_tracked_video_matting_eval.py`
- Create: `tests/test_run_tracked_video_matting_eval.py`
- Reuse: `modnet_matting_service.py`, `matanyone_service.py`, `subject_edge_refine.py`

- [ ] **Step 1: Write tests for output-frame selection, constraint mapping, and metrics**

```python
from run_tracked_video_matting_eval import (
    select_output_frames,
    apply_tracked_alpha_constraints,
    compute_tracked_metrics,
    compute_edge_temporal_jitter,
)


def test_select_output_frames_uses_configured_four_indices():
    selected = select_output_frames(list(range(16)), [3, 7, 10, 13])
    assert selected == [3, 7, 10, 13]


def test_apply_tracked_constraints_clears_visitors_and_preserves_subject_core():
    frame = tracked_frame_result()
    raw = np.full((20, 20), 120, dtype=np.uint8)
    image = apply_tracked_alpha_constraints(Image.new("RGB", (20, 20)), raw, frame)
    alpha = np.array(image.getchannel("A"))
    assert np.all(alpha[frame.sure_background] == 0)
    assert np.all(alpha[frame.sure_foreground] == 255)


def test_tracked_metrics_report_visitor_core_outside_and_edge_jitter():
    frame = tracked_frame_result()
    image = constrained_image(frame)
    metrics = compute_tracked_metrics(image, frame)
    assert metrics["visitor_track_alpha_ratio"] == 0
    assert metrics["subject_core_missing_ratio"] == 0
    assert metrics["outside_subject_soft_alpha_ratio"] >= 0
    assert compute_edge_temporal_jitter([image, image], [frame, frame]) == 0
```

- [ ] **Step 2: Run tests and observe missing-runner failure**

Run: `python -m pytest tests/test_run_tracked_video_matting_eval.py -q --basetemp=.pytest_tmp/tracked_eval_metrics_red`

Expected: FAIL because the runner module does not exist.

- [ ] **Step 3: Implement reusable runner primitives**

Add the following functions to `run_tracked_video_matting_eval.py`:

```python
def select_output_frames(frames: list[Any], output_indices: list[int]) -> list[Any]:
    if len(output_indices) != 4 or any(index < 0 or index >= len(frames) for index in output_indices):
        raise ValueError("Exactly four valid output frame indices are required.")
    return [frames[index] for index in output_indices]


def apply_tracked_alpha_constraints(image: Image.Image, raw_alpha: np.ndarray,
                                    frame: TrackedFrameResult) -> Image.Image:
    # sure_background alpha = 0; sure_foreground alpha = 255; unknown retains raw alpha.


def compute_tracked_metrics(image: Image.Image, frame: TrackedFrameResult) -> dict[str, float]:
    # visitor_track_alpha_ratio, subject_core_missing_ratio,
    # outside_subject_soft_alpha_ratio, foreground_px.


def compute_edge_temporal_jitter(images: list[Image.Image],
                                 frames: list[TrackedFrameResult]) -> float:
    # Align adjacent alpha masks by selected bbox center translation.
    # Measure XOR of 1px Canny boundary bands normalized by union boundary pixels.
```

Add JSON aggregation for mean/max metrics and elapsed seconds. Keep these helpers independent of camera and real neural models so tests stay deterministic.

- [ ] **Step 4: Run metric tests**

Run: `python -m pytest tests/test_run_tracked_video_matting_eval.py -q --basetemp=.pytest_tmp/tracked_eval_metrics_green`

Expected: PASS.

- [ ] **Step 5: Commit runner primitives and tests**

```powershell
git add -- run_tracked_video_matting_eval.py tests/test_run_tracked_video_matting_eval.py
git commit -m "feat: add tracked video matting evaluation metrics"
```

### Task 4: Three-Branch Offline Runner

**Files:**
- Modify: `run_tracked_video_matting_eval.py`
- Modify: `tests/test_run_tracked_video_matting_eval.py`

- [ ] **Step 1: Write an integration test with fake tracker and fake matting backends**

```python
def test_run_take_generates_three_branches_and_four_final_outputs(tmp_path):
    take = create_fake_burst_take(tmp_path, frame_count=16, output_indices=[3, 7, 10, 13])
    tracker = FakeSequenceTracker(ok_sequence(16))
    matanyone = FakeVideoMatting(alpha_frames=alpha_sequence(16))
    modnet = FakeModnet()

    summary = run_take(
        take,
        tmp_path / "out",
        tracker=tracker,
        matanyone=matanyone,
        modnet=modnet,
        compose_final=lambda image, *_: image,
    )

    assert summary["status"] == "ok"
    assert len(list((tmp_path / "out" / "tracked_matanyone" / "final").glob("*.jpg"))) == 4
    assert len(list((tmp_path / "out" / "tracked_modnet_4frame" / "final").glob("*.jpg"))) == 4
    assert (tmp_path / "out" / "sheets" / "final_sheet.jpg").exists()


def test_run_take_records_tracking_failure_without_invoking_matanyone(tmp_path):
    tracker = FakeSequenceTracker(failed_sequence())
    matanyone = FakeVideoMatting(alpha_frames=[])
    summary = run_take(create_fake_burst_take(tmp_path), tmp_path / "out", tracker=tracker, matanyone=matanyone)
    assert summary["status"] == "tracking_failed"
    assert matanyone.calls == 0
```

- [ ] **Step 2: Run the integration tests and observe failure**

Run: `python -m pytest tests/test_run_tracked_video_matting_eval.py -q --basetemp=.pytest_tmp/tracked_eval_runner_red`

Expected: FAIL because `run_take()` and branch output behavior are not yet implemented.

- [ ] **Step 3: Implement `run_take()` and the CLI**

Implement the full take flow:

```python
def run_take(manifest_path: Path, output_dir: Path, *,
             tracker: SubjectInstanceTracker,
             matanyone: MatAnyoneService | None,
             modnet: ModnetMattingService | None,
             compose_final=_compose_final) -> dict[str, Any]:
    # Read manifest and 16 frames.
    # Track all frames; abort neural branches on tracking_failed.
    # Branch 1: tracked_yolo_seg_mask for the configured four output frames.
    # Branch 2: tracked_modnet_4frame on the same four output frames.
    # Branch 3: tracked_matanyone processes all frames, then saves the same four.
    # Apply tracked constraints -> edge refine -> reapply tracked constraints.
    # Save cutouts/final/sheets/debug and metrics summary.
```

CLI:

```powershell
python run_tracked_video_matting_eval.py `
  --input-root generated/burst_eval_inputs/<session> `
  --output-root generated/tracked_video_matting_eval/<timestamp> `
  --yolo-seg-model-path models/yolo11x-seg.pt `
  --include-modnet `
  --include-matanyone
```

Implementation rules:

- Pass the complete ordered `frames/` directory to `MatAnyoneService.process_video()` and use the first tracked subject mask as its prompt.
- Convert each `TrackedFrameResult` to the existing `InstanceSegmentationResult` shape only where existing MODNet/edge helper APIs require it; never re-run independent subject selection for output frames.
- Use `SubjectEdgeRefineConfig.from_mapping()` from current config for all three branches; enforce tracked foreground/background again after refinement.
- Build final sheets with exactly four columns and rows for available branches.
- Record model unavailability or per-take failure in JSON instead of aborting the entire session.

- [ ] **Step 4: Run integration and existing matting tests**

Run:

```powershell
python -m pytest tests/test_run_tracked_video_matting_eval.py tests/test_modnet_matting_service.py tests/test_matanyone_service.py -q --basetemp=.pytest_tmp/tracked_eval_runner_green
```

Expected: PASS.

- [ ] **Step 5: Commit runner functionality**

```powershell
git add -- run_tracked_video_matting_eval.py tests/test_run_tracked_video_matting_eval.py
git commit -m "feat: evaluate tracked burst video matting offline"
```

### Task 5: Regression Verification And Real Burst Collection

**Files:**
- Generated only: `generated/burst_eval_inputs/`, `generated/tracked_video_matting_eval/`
- No online source modifications.

- [ ] **Step 1: Run the complete test suite before hardware validation**

Run: `python -m pytest tests -q --basetemp=.pytest_tmp/tracked_video_full`

Expected: all tests pass; existing Pillow deprecation warnings may remain.

- [ ] **Step 2: Collect at least twelve real burst takes**

Run each scenario twice while keeping the subject position representative of kiosk use:

```powershell
python capture_burst_eval.py --scenario single_subject --take take_01
python capture_burst_eval.py --scenario single_subject --take take_02
python capture_burst_eval.py --scenario visitor_behind_left --take take_01
python capture_burst_eval.py --scenario visitor_behind_left --take take_02
python capture_burst_eval.py --scenario visitor_behind_right --take take_01
python capture_burst_eval.py --scenario visitor_behind_right --take take_02
python capture_burst_eval.py --scenario visitor_crossing --take take_01
python capture_burst_eval.py --scenario visitor_crossing --take take_02
python capture_burst_eval.py --scenario visitor_close_contact --take take_01
python capture_burst_eval.py --scenario visitor_close_contact --take take_02
python capture_burst_eval.py --scenario hair_edge_motion --take take_01
python capture_burst_eval.py --scenario hair_edge_motion --take take_02
```

Expected: each manifest contains at least `12` valid frames and status `ok`.

- [ ] **Step 3: Run the real offline evaluation**

```powershell
python run_tracked_video_matting_eval.py `
  --input-root generated/burst_eval_inputs/<session> `
  --output-root generated/tracked_video_matting_eval/<timestamp> `
  --yolo-seg-model-path models/yolo11x-seg.pt `
  --include-modnet `
  --include-matanyone
```

Expected: every valid take produces three branch directories where models are available, four final posters per branch, per-take sheets, and a session summary JSON.

- [ ] **Step 4: Review quality evidence and record the decision**

Review, per scenario, `final_sheet.jpg` and `cutout_sheet.jpg`. In the session JSON confirm:

```text
visitor_track_alpha_ratio
subject_core_missing_ratio
outside_subject_soft_alpha_ratio
edge_temporal_jitter
track_switch_count
track_lost_frames
elapsed_seconds
```

Decision rule:

- Advance `tracked_matanyone` only if it has no visually identifiable visitor remnants and improves edge softness/jitter over the existing four-frame MatAnyone result without visible subject loss.
- If `tracked_modnet_4frame` is equal or better visually, stop expanding MatAnyone for online use and select tracked constraints + MODNet for the next design cycle.

- [ ] **Step 5: Do not commit generated media; commit implementation only if requested**

Run: `git status --short`

Expected: `generated/` outputs remain unstaged. Do not stage model weights, vendor repositories, videos, or current unrelated dirty files.

## Verification Checklist

- The implementation never changes the online selection or default capture pipeline.
- A single offline take consumes `12–16` real sequential frames and emits four comparable final images per available branch.
- The tracked subject is selected once, then maintained by track id with bounded IoU recovery.
- Visitor masks become sure background in every evaluated frame.
- Full automated tests pass before collecting or interpreting visual results.
- Real-session output contains enough debug data to explain tracking failures rather than hiding them behind a final image.
