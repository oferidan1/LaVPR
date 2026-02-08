from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FileLock:
    """
    Very small cross-process lock using `fcntl.flock` on a lock file.

    This prevents two concurrent pipeline runs from writing the same output CSVs
    (which can otherwise produce duplicated back-to-back rows).
    """

    target_path: Path
    timeout_sec: float = 10.0
    poll_interval_sec: float = 0.2

    _fh: Optional[object] = None

    @property
    def lock_path(self) -> Path:
        return Path(str(self.target_path) + ".lock")

    def acquire(self) -> None:
        # Import lazily: only available on Unix (this repo runs on Linux/WSL/HPC).
        import fcntl  # type: ignore

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = self.lock_path.open("a+", encoding="utf-8")
        self._fh = fh

        deadline = time.time() + float(self.timeout_sec)
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fh.seek(0)
                fh.truncate()
                fh.write(
                    f"pid={os.getpid()}\n"
                    f"target={self.target_path}\n"
                    f"acquired_at={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                )
                fh.flush()
                return
            except BlockingIOError:
                if time.time() >= deadline:
                    raise SystemExit(
                        f"[lock] Timed out waiting for lock: {self.lock_path}\n"
                        f"[lock] Another process is likely writing: {self.target_path}\n"
                        f"[lock] If this is a stale lock, delete: {self.lock_path}"
                    )
                time.sleep(float(self.poll_interval_sec))

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            import fcntl  # type: ignore

            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass
        self._fh = None

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
