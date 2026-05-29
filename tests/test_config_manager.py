"""Test config_manager: deep merge, defaults, mojibake, config I/O."""

import json
from unittest.mock import patch

from config_manager import (
    DEFAULT_CONFIG,
    deep_merge,
    load_config,
    normalize_mojibake_text,
    normalize_text_tree,
    save_config,
)


class TestDeepMerge:
    def test_empty_override_returns_base(self):
        result = deep_merge({"a": 1}, {})
        assert result == {"a": 1}

    def test_simple_override(self):
        result = deep_merge({"a": 1, "b": 2}, {"b": 99})
        assert result == {"a": 1, "b": 99}

    def test_nested_merge(self):
        base = {"outer": {"inner_a": 1, "inner_b": 2}, "top": "x"}
        override = {"outer": {"inner_b": 42}}
        result = deep_merge(base, override)
        assert result["outer"]["inner_a"] == 1
        assert result["outer"]["inner_b"] == 42
        assert result["top"] == "x"

    def test_new_key_added(self):
        result = deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_override_scalar_with_dict(self):
        result = deep_merge({"a": 1}, {"a": {"nested": True}})
        assert result == {"a": {"nested": True}}


class TestNormalizeMojibake:
    def test_normal_text_passes_through(self):
        assert normalize_mojibake_text("人在阵地在") == "人在阵地在"

    def test_empty_string(self):
        assert normalize_mojibake_text("") == ""

    def test_none_returns_none(self):
        assert normalize_mojibake_text(None) is None  # type: ignore[arg-type]

    def test_latin1_mojibake_fixed(self):
        original = "人在阵地在"
        broken = original.encode("utf-8").decode("latin1")
        assert broken != original
        fixed = normalize_mojibake_text(broken)
        assert fixed == original


class TestNormalizeTextTree:
    def test_string(self):
        assert normalize_text_tree("hello") == "hello"

    def test_list_of_strings(self):
        result = normalize_text_tree(["a", "b"])
        assert result == ["a", "b"]

    def test_nested_dict(self):
        original = "人在阵地在"
        broken = original.encode("utf-8").decode("latin1")
        result = normalize_text_tree({"slogan": broken})
        assert result == {"slogan": original}

    def test_deeply_nested(self):
        original = "中文"
        broken = original.encode("utf-8").decode("latin1")
        result = normalize_text_tree({"a": [{"b": broken}]})
        assert result == {"a": [{"b": original}]}


class TestDefaultConfig:
    def test_has_required_keys(self):
        for key in ["server", "autostart", "camera", "rotation", "background_set", "output",
                     "text_style", "text_tuning", "laser_trigger", "matting_api",
                     "subject_locator", "subject_alpha_filter", "subject_visitor_suppression",
                     "subject_edge_refine", "temporal_subject_fusion"]:
            assert key in DEFAULT_CONFIG

    def test_autostart_defaults(self):
        assert DEFAULT_CONFIG["autostart"]["enabled"] is False
        assert DEFAULT_CONFIG["autostart"]["method"] == "startup_folder"
        assert DEFAULT_CONFIG["autostart"]["task_name"] == "HuaitaTextKiosk"
        assert DEFAULT_CONFIG["autostart"]["delay_seconds"] == 10
        assert DEFAULT_CONFIG["autostart"]["run_level"] == "LIMITED"
        assert DEFAULT_CONFIG["autostart"]["startup_args"] == []

    def test_camera_logging_defaults(self):
        assert DEFAULT_CONFIG["camera"]["log_enabled"] is True
        assert DEFAULT_CONFIG["camera"]["log_path"] == ""
        assert DEFAULT_CONFIG["camera"]["stale_frame_seconds"] == 5.0

    def test_laser_watchdog_defaults(self):
        laser = DEFAULT_CONFIG["laser_trigger"]
        assert laser["keepalive_enabled"] is True
        assert laser["keepalive_no_data_seconds"] == 3.0
        assert laser["reconnect_no_data_seconds"] == 8.0

    def test_subject_locator_defaults(self):
        locator = DEFAULT_CONFIG["subject_locator"]
        assert locator["enabled"] is True
        assert locator["provider"] == "yolo_person_bbox"
        assert locator["model_path"] == "models/yolo11x.pt"
        assert locator["min_confidence"] == 0.45
        assert locator["roi_expand_ratio"] == 0.12
        assert locator["min_person_height_ratio"] == 0.25
        assert locator["roi_side_trim_enabled"] is True
        assert locator["roi_side_trim_margin_ratio"] == 0.08
        assert locator["roi_side_trim_max_overlap_ratio"] == 0.20

    def test_subject_alpha_filter_defaults(self):
        alpha_filter = DEFAULT_CONFIG["subject_alpha_filter"]
        assert alpha_filter["enabled"] is True
        assert alpha_filter["mode"] == "strong_remove_visitors"
        assert alpha_filter["alpha_threshold"] == 8
        assert alpha_filter["subject_box_expand_ratio"] == 0.08
        assert alpha_filter["visitor_box_expand_ratio"] == 0.18
        assert alpha_filter["keep_nearby_component_px"] == 6
        assert alpha_filter["debug_enabled"] is True

    def test_subject_visitor_suppression_defaults(self):
        suppression = DEFAULT_CONFIG["subject_visitor_suppression"]
        assert suppression["enabled"] is True
        assert suppression["pre_aliyun_enabled"] is True
        assert suppression["post_alpha_hard_clear"] is True
        assert suppression["visitor_preclean_expand_ratio"] == 0.18
        assert suppression["subject_protect_expand_ratio"] == 0.04
        assert suppression["fill_mode"] == "inpaint"
        assert suppression["inpaint_radius"] == 9
        assert suppression["debug_enabled"] is True

    def test_subject_edge_refine_defaults(self):
        refine = DEFAULT_CONFIG["subject_edge_refine"]
        assert refine["enabled"] is True
        assert refine["min_component_area_ratio"] == 0.0012
        assert refine["open_kernel_px"] == 1
        assert refine["feather_radius_px"] == 1.3
        assert refine["edge_ring_blur_enabled"] is True
        assert refine["edge_ring_inner_px"] == 3
        assert refine["edge_ring_outer_px"] == 3
        assert refine["edge_ring_sigma"] == 1.0
        assert refine["arm_edge_tighten_enabled"] is True
        assert refine["arm_edge_tighten_px"] == 1
        assert refine["arm_edge_tighten_strength"] == "medium"
        assert refine["hard_clear_feather_px"] == 4
        assert refine["effective_bbox_alpha_threshold"] == 16
        assert refine["debug_enabled"] is True

    def test_temporal_subject_fusion_defaults(self):
        fusion = DEFAULT_CONFIG["temporal_subject_fusion"]
        assert fusion["enabled"] is True
        assert fusion["mode"] == "alpha_stability_fusion"
        assert fusion["min_frames"] == 4
        assert fusion["alignment_mode"] == "ecc_translation"
        assert fusion["alpha_vote_threshold"] == 0.64
        assert fusion["edge_consistency_weight"] == 0.42
        assert fusion["noise_component_min_area_ratio"] == 0.0012
        assert fusion["fallback_to_single"] is True
        assert fusion["debug_enabled"] is True

    def test_output_dimensions(self):
        assert DEFAULT_CONFIG["output"]["width"] == 1080
        assert DEFAULT_CONFIG["output"]["height"] == 1920

    def test_person_layout_has_width_limit(self):
        layout = DEFAULT_CONFIG["person_layout"]
        assert layout["max_width_ratio"] == 0.92

    def test_server_port_fallback_defaults(self):
        assert DEFAULT_CONFIG["server"]["auto_port_fallback"] is True
        assert DEFAULT_CONFIG["server"]["port_fallback_attempts"] == 10

    def test_background_set_has_four_items(self):
        assert len(DEFAULT_CONFIG["background_set"]["items"]) == 4


class TestConfigIO:
    def test_load_config_creates_file_if_missing(self, temp_dir):
        config_path = temp_dir / "config.json"
        with patch("config_manager._config_path", return_value=config_path):
            cfg = load_config()
            assert config_path.exists()
            assert "server" in cfg
            assert cfg["rotation"]["rotation_start_time"] > 0

    def test_save_and_reload(self, temp_dir):
        config_path = temp_dir / "config.json"
        with patch("config_manager._config_path", return_value=config_path):
            cfg = load_config()
            cfg["server"]["port"] = 9999
            save_config(cfg)

            reloaded = load_config()
            assert reloaded["server"]["port"] == 9999

    def test_load_merges_partial_config(self, temp_dir):
        config_path = temp_dir / "config.json"
        partial = {"server": {"port": 8888}, "rotation": {"slogans": ["自定义标语"]}}
        config_path.write_text(json.dumps(partial, ensure_ascii=False), encoding="utf-8")

        with patch("config_manager._config_path", return_value=config_path):
            cfg = load_config()
            assert cfg["server"]["port"] == 8888
            assert cfg["server"]["host"] == "127.0.0.1"  # default preserved
            assert cfg["rotation"]["slogans"] == ["自定义标语"]

    def test_load_backs_up_invalid_json_and_restores_defaults(self, temp_dir):
        config_path = temp_dir / "config.json"
        config_path.write_text("{ invalid json", encoding="utf-8")

        with patch("config_manager._config_path", return_value=config_path):
            cfg = load_config()

        backups = list(temp_dir.glob("config.invalid-*.json"))
        assert backups
        assert config_path.exists()
        assert cfg["server"]["host"] == DEFAULT_CONFIG["server"]["host"]
        assert "_config_warning" in cfg

    def test_save_config_uses_temp_file_cleanup(self, temp_dir):
        config_path = temp_dir / "config.json"
        with patch("config_manager._config_path", return_value=config_path):
            save_config({"server": {"port": 7777}})

        assert config_path.exists()
        assert not config_path.with_name("config.json.tmp").exists()
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        assert saved["server"]["port"] == 7777
