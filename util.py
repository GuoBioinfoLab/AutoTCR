"""Compatibility utilities for original scripts; inference uses validated settings."""
import math
import yaml


class AttrDict(dict):
    """Dictionary exposing nested keys through attributes."""
    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return AttrDict(value) if isinstance(value, dict) else value


def load_config(config_file):
    with open(config_file, encoding="utf-8") as reader:
        config = yaml.safe_load(reader)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping.")
    return AttrDict(config)


def min_power_greater_than(value, base=2):
    """Return the smallest power of base greater than or equal to value."""
    return math.pow(base, math.ceil(math.log(value, base)))


def insert_whitespace(seq):
    if isinstance(seq, str):
        return " ".join(seq)
    return [" ".join(item) for item in seq]
