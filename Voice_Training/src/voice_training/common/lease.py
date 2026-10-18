"""CLI task lock: acquire a Windows byte lock; release it on normal exit or crash."""
from pathlib import Path


class InferenceLease:
    """An OS file lock; crashes release it automatically and stale files are harmless."""
    def __init__(self, path):
        """Initialize the instance state and resources owned by this component."""
        self.path = Path(path)
        self.stream = None

    def acquire(self):
        """Acquire a nonblocking Windows file lock or report an active project task."""
        import msvcrt
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open('a+b')
        if self.path.stat().st_size == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            stream.close()
            raise RuntimeError('本项目已有任务正在运行，请等待或停止该任务后重试。') from None
        self.stream = stream

    def close(self):
        """Release and close the held Windows lock, if any."""
        if self.stream is not None:
            import msvcrt
            self.stream.seek(0)
            try:
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                self.stream.close()
                self.stream = None
