import numpy as np
from PIL import Image

from subject_temporal_fusion import TemporalSubjectFusionConfig, fuse_subjects_temporally


def _subject_with_rect_and_noise(
    *,
    size=(120, 120),
    rect=(30, 20, 90, 105),
    noise=None,
):
    arr = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    arr[:, :, :3] = 140
    x1, y1, x2, y2 = rect
    arr[y1:y2, x1:x2, 3] = 255
    if noise:
        nx1, ny1, nx2, ny2 = noise
        arr[ny1:ny2, nx1:nx2, 3] = 255
    return Image.fromarray(arr, "RGBA")


def test_temporal_fusion_removes_single_frame_noise():
    frames = [
        _subject_with_rect_and_noise(noise=(4, 4, 8, 8)),
        _subject_with_rect_and_noise(),
        _subject_with_rect_and_noise(),
        _subject_with_rect_and_noise(),
    ]
    cfg = TemporalSubjectFusionConfig(edge_consistency_weight=0.0, alpha_vote_threshold=0.6)

    fused, report = fuse_subjects_temporally(frames, cfg)

    assert report.fallback_reason is None
    assert fused[0].getpixel((5, 5))[3] == 0
    assert fused[0].getpixel((60, 60))[3] > 0


def test_temporal_fusion_suppresses_two_of_four_transient_region():
    frames = [
        _subject_with_rect_and_noise(noise=(90, 10, 112, 45)),
        _subject_with_rect_and_noise(noise=(90, 10, 112, 45)),
        _subject_with_rect_and_noise(),
        _subject_with_rect_and_noise(),
    ]
    cfg = TemporalSubjectFusionConfig(edge_consistency_weight=0.0, alpha_vote_threshold=0.6)

    fused, report = fuse_subjects_temporally(frames, cfg)

    assert report.fallback_reason is None
    assert fused[1].getpixel((100, 25))[3] <= 2
    assert fused[1].getpixel((60, 60))[3] > 0


def test_temporal_fusion_preserves_stable_subject_region():
    frames = [_subject_with_rect_and_noise() for _ in range(4)]
    cfg = TemporalSubjectFusionConfig(edge_consistency_weight=0.0, alpha_vote_threshold=0.6)

    fused, report = fuse_subjects_temporally(frames, cfg)

    assert report.fallback_reason is None
    for frame in fused:
        assert frame.getpixel((60, 60))[3] > 0
        assert frame.getpixel((10, 10))[3] == 0


def test_temporal_fusion_preserves_opaque_alpha_for_stable_subject_core():
    frames = [_subject_with_rect_and_noise() for _ in range(4)]
    cfg = TemporalSubjectFusionConfig(edge_consistency_weight=0.0, alpha_vote_threshold=0.6)

    fused, report = fuse_subjects_temporally(frames, cfg)

    assert report.fallback_reason is None
    for frame in fused:
        assert frame.getpixel((60, 60))[3] >= 250


def test_temporal_fusion_edge_weight_does_not_remove_stable_interior():
    frames = [_subject_with_rect_and_noise() for _ in range(4)]
    cfg = TemporalSubjectFusionConfig(edge_consistency_weight=0.42, alpha_vote_threshold=0.64)

    fused, report = fuse_subjects_temporally(frames, cfg)

    assert report.fallback_reason is None
    for frame in fused:
        assert frame.getpixel((60, 60))[3] >= 250


def test_temporal_fusion_falls_back_when_insufficient_frames():
    frames = [_subject_with_rect_and_noise() for _ in range(3)]
    cfg = TemporalSubjectFusionConfig(min_frames=4, fallback_to_single=True)

    fused, report = fuse_subjects_temporally(frames, cfg)

    assert report.fallback_reason == "insufficient_frames"
    assert [img.tobytes() for img in fused] == [img.tobytes() for img in frames]
