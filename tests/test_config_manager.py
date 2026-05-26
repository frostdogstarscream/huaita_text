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
                     "text_style", "text_tuning", "laser_trigger", "matting_api"]:
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

    def test_output_dimensions(self):
        assert DEFAULT_CONFIG["output"]["width"] == 1080
        assert DEFAULT_CONFIG["output"]["height"] == 1920

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
