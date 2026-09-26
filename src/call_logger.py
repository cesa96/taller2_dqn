import csv
from datetime import datetime
from pathlib import Path


_LOG_PATH = Path(__file__).resolve().parents[1] / "results" / "method_calls.csv"


def log_method_call(method_name: str) -> None:
    """Append a method name and its local timestamp to the calls CSV."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not _LOG_PATH.exists() or _LOG_PATH.stat().st_size == 0

    with _LOG_PATH.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if write_header:
            writer.writerow(("metodo", "fecha_hora"))
        writer.writerow(
            (method_name, datetime.now().astimezone().isoformat(timespec="seconds"))
        )
