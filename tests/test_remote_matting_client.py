import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from remote_matting_client import RemoteTrackedMattingClient


def test_remote_client_process_sequence_success():
    cfg = {
        "remote_matting": {
            "enabled": True,
            "base_url": "http://127.0.0.1:18080",
            "job_timeout_seconds": 2.0,
            "poll_interval_seconds": 0.01,
        }
    }
    client = RemoteTrackedMattingClient(cfg)
    base_tmp = Path(tempfile.mkdtemp(prefix="remote_client_test_", dir="."))
    sequence_dir = base_tmp / "task_tracked"
    frames_dir = sequence_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_path = sequence_dir / "burst.avi"
    video_path.write_bytes(b"fake-video")
    frame_paths = []
    for idx in range(4):
        path = frames_dir / f"{idx+1:06d}.jpg"
        path.write_bytes(b"jpg")
        frame_paths.append(path)

    sequence_capture = {
        "video_path": video_path,
        "frame_paths": frame_paths,
        "output_indices": [0, 1, 2, 3],
        "shot_task_ids": ["a", "b", "c", "d"],
    }

    post_resp = MagicMock(status_code=200)
    post_resp.json.return_value = {"job_id": "job_1"}
    poll_resp = MagicMock(status_code=200)
    poll_resp.json.return_value = {
        "status": "completed",
        "results": [
            {"image_id": "i1", "background_id": "bg", "background_name": "bg", "remote_image_url": "/api/remote/jobs/job_1/final/1"},
            {"image_id": "i2", "background_id": "bg", "background_name": "bg", "remote_image_url": "/api/remote/jobs/job_1/final/2"},
            {"image_id": "i3", "background_id": "bg", "background_name": "bg", "remote_image_url": "/api/remote/jobs/job_1/final/3"},
            {"image_id": "i4", "background_id": "bg", "background_name": "bg", "remote_image_url": "/api/remote/jobs/job_1/final/4"},
        ],
    }
    img_resp = MagicMock(status_code=200, content=b"jpeg")

    with patch("remote_matting_client.requests.post", return_value=post_resp), \
         patch("remote_matting_client.requests.get", side_effect=[poll_resp, img_resp, img_resp, img_resp, img_resp]):
        results, metrics = client.process_sequence(
            task_id="task_1",
            sequence_capture=sequence_capture,
            slogan="test",
            slogan_row=1,
            background_item={"id": "bg", "name": "bg", "path": "x"},
        )

    assert len(results) == 4
    assert metrics["remote_job_id"] == "job_1"
