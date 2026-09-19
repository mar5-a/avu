import os
import yaml
import argparse
from pathlib import Path

from dotenv import load_dotenv


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_config(config: dict, args: argparse.Namespace):
    for key, value in args.__dict__.items():
        if key in config and value is not None:
            config[key] = value
    config["api_base"] = os.getenv(config["api_base"], None)
    config["api_key"] = os.getenv(config["api_key"], None)
    config["api_version"] = os.getenv(config["api_version"], None)
    return config


# General config
CONFIG_DIR = Path(__file__).resolve().parent

# Load a .env from the nearest enclosing directory, walking up from this package
# to the project root. load_dotenv does not override variables already set in the
# shell, so an exported key still wins. Searching beats a fixed relative path here
# because load_dotenv silently does nothing when the path is wrong.
for _parent in (CONFIG_DIR, *CONFIG_DIR.parents):
    if (_parent / ".env").is_file():
        load_dotenv(_parent / ".env")
        break

general_config = load_config(str(CONFIG_DIR / "general.yaml"))

prompts_config = load_config(str(CONFIG_DIR / "prompts.yaml"))