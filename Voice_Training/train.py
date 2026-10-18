"""Python entry: add the local source package -> configure runtime -> CLI.main.

Input is CLI arguments with project-relative config/data paths. Training stages
are selected explicitly; importing this file never starts training.
"""
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(SOURCE))

from voice_training.common.runtime import configure_runtime

configure_runtime()

if __name__ == '__main__':
    from voice_training.cli import main
    raise SystemExit(main())
