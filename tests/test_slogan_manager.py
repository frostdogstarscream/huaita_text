"""Test slogan_manager: display, draw, row detection, rotation, text tuning."""

import pytest

from slogan_manager import (
    _normalize_slogan_lookup_key,
    get_rotation_snapshot,
    get_slogan_snapshot_by_sequence_no,
    normalize_slogan_lines_to_row_count,
    resolve_text_tuning,
    set_rotation_to_index,
    slogan_display_text,
    slogan_draw_text,
    slogan_explicit_lines,
    slogan_row_from_entry,
)


class TestSloganDisplayText:
    def test_string_entry(self):
        assert slogan_display_text("人在阵地在") == "人在阵地在"

    def test_dict_entry(self):
        entry = {"content": "人在阵地在", "row": 1}
        assert slogan_display_text(entry) == "人在阵地在"

    def test_multiline_dict_collapses_to_single_line(self):
        entry = {"content": "我能挺住\n不要说我负了伤", "row": 2}
        result = slogan_display_text(entry)
        assert "\n" not in result
        assert " " in result

    def test_empty_entry(self):
        assert slogan_display_text("") == ""


class TestSloganDrawText:
    def test_preserves_line_breaks(self):
        entry = {"content": "我能挺住\n不要说我负了伤", "row": 2}
        result = slogan_draw_text(entry)
        assert "\n" in result

    def test_string_entry_returns_trimmed(self):
        assert slogan_draw_text("  标语  ") == "标语"


class TestSloganExplicitLines:
    def test_no_newlines_returns_none(self):
        assert slogan_explicit_lines("单行标语") is None

    def test_with_newlines(self):
        lines = slogan_explicit_lines("第一行\n第二行")
        assert lines == ["第一行", "第二行"]

    def test_empty_lines_filtered(self):
        lines = slogan_explicit_lines("第一行\n\n第二行\n")
        assert lines == ["第一行", "第二行"]


class TestSloganRowFromEntry:
    def test_dict_with_explicit_row(self):
        assert slogan_row_from_entry({"content": "x", "row": 3}) == 3

    def test_dict_infers_from_newlines(self):
        entry = {"content": "a\nb\nc"}
        assert slogan_row_from_entry(entry) == 3

    def test_string_entry_returns_1(self):
        assert slogan_row_from_entry("任意标语") == 1

    def test_invalid_row_clamped(self):
        assert slogan_row_from_entry({"content": "x", "row": -5}) == 1


class TestNormalizeSloganLinesToRowCount:
    def test_fewer_lines_than_target(self):
        result = normalize_slogan_lines_to_row_count(["单行"], 2)
        assert result == ["单行"]

    def test_more_lines_merged_to_target_2(self):
        result = normalize_slogan_lines_to_row_count(["a", "b", "c"], 2)
        assert result == ["a", "b c"]

    def test_more_lines_merged_to_target_3(self):
        result = normalize_slogan_lines_to_row_count(["a", "b", "c", "d"], 3)
        assert result == ["a", "b", "c d"]

    def test_target_1_joins_all(self):
        result = normalize_slogan_lines_to_row_count(["a", "b", "c"], 1)
        assert result == ["a b c"]


class TestRotationSnapshot:
    def test_returns_slogan_from_config(self, patched_app_state):
        snapshot = get_rotation_snapshot()
        assert "slogan" in snapshot
        assert "slogan_content" in snapshot
        assert "index" in snapshot
        assert "seconds_to_next" in snapshot

    def test_rotation_index_advances(self, patched_app_state):
        slogans = patched_app_state["config"]["rotation"]["slogans"]
        assert len(slogans) >= 3

    def test_empty_slogans_falls_back(self, patched_app_state):
        saved = patched_app_state["config"]["rotation"]["slogans"]
        patched_app_state["config"]["rotation"]["slogans"] = []
        try:
            snapshot = get_rotation_snapshot()
            assert snapshot["slogan"] == "欢迎来到互动拍照区"
        finally:
            patched_app_state["config"]["rotation"]["slogans"] = saved


class TestSetRotationToIndex:
    def test_valid_index(self, patched_app_state):
        result = set_rotation_to_index(0)
        assert result["index"] == 0

    def test_invalid_index_raises(self, patched_app_state):
        with pytest.raises(ValueError, match="out of range"):
            set_rotation_to_index(999)


class TestResolveTextTuning:
    def test_returns_defaults(self, patched_app_state):
        tuning = resolve_text_tuning("某个标语")
        assert "preferred_lines" in tuning
        assert "font_scale" in tuning
        assert "line_priority" in tuning

    def test_forced_lines_are_stripped(self, patched_app_state):
        patched_app_state["config"]["text_tuning"]["by_slogan"]["测试标语"] = {
            "forced_lines": ["测试", "标语"],
            "font_scale": 1.2,
        }
        tuning = resolve_text_tuning("测试标语")
        assert "forced_lines" not in tuning
        assert tuning["font_scale"] == 1.2

    def test_per_slogan_override(self, patched_app_state):
        patched_app_state["config"]["text_tuning"]["by_slogan"]["特殊标语"] = {
            "y_offset": 10,
            "preferred_lines": 1,
        }
        tuning = resolve_text_tuning("特殊标语")
        assert tuning["y_offset"] == 10
        assert tuning["preferred_lines"] == 1


class TestNormalizeSloganLookupKey:
    def test_collapses_whitespace(self):
        result = _normalize_slogan_lookup_key("只要还有一个人活着\n就要守住阵地")
        assert "\n" not in result
        assert "  " not in result


class TestGetSloganSnapshotBySequenceNo:
    def test_returns_first_slogan(self, patched_app_state):
        snapshot = get_slogan_snapshot_by_sequence_no(1)
        assert snapshot["sequence_no"] == 1
        assert snapshot["slogan"] == "测试标语一"
        assert snapshot["index"] == 0

    def test_returns_last_slogan(self, patched_app_state):
        snapshot = get_slogan_snapshot_by_sequence_no(3)
        assert snapshot["sequence_no"] == 3
        assert snapshot["index"] == 2

    def test_returns_multiline_slogan(self, patched_app_state):
        snapshot = get_slogan_snapshot_by_sequence_no(3)
        assert snapshot["slogan_row"] == 2

    def test_raises_for_zero(self, patched_app_state):
        with pytest.raises(ValueError, match="out of range"):
            get_slogan_snapshot_by_sequence_no(0)

    def test_raises_for_exceeding_count(self, patched_app_state):
        with pytest.raises(ValueError, match="out of range"):
            get_slogan_snapshot_by_sequence_no(4)

    def test_does_not_modify_rotation_start_time(self, patched_app_state):
        original = patched_app_state["config"]["rotation"]["rotation_start_time"]
        get_slogan_snapshot_by_sequence_no(2)
        assert patched_app_state["config"]["rotation"]["rotation_start_time"] == original
