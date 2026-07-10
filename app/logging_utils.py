import json
import logging
import sys
import time


def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger("gateway")
    root.setLevel(logging.INFO)
    root.addHandler(handler)


def log_event(**fields):
    """
    Emits a single-line JSON log record — ready to ship to a SIEM
    (Datadog/Splunk/Sentinel) later. Never log full prompt content or
    full API keys here; only metadata.
    """
    record = {"ts": time.time(), **fields}
    logging.getLogger("gateway").info(json.dumps(record))
