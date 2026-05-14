import os
import time
from pathlib import Path


def current_local_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def daily_history_path(summary_path):
    summary_path = Path(summary_path)
    history_dir = summary_path.parent / "history"
    return history_dir / f"{summary_path.stem}-{time.strftime('%Y-%m-%d')}{summary_path.suffix}"


def active_history_label(summary_path):
    history_path = daily_history_path(summary_path)
    return (Path("history") / history_path.name).as_posix()


def atomic_write_text(file_path, content):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    os.replace(temp_path, file_path)


def append_line(file_path, line):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(line if line.endswith("\n") else f"{line}\n")
