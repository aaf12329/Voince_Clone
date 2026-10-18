"""Worker entry: main -> shared InferenceEngine.generate -> event JSON.
UTF-8 is enforced even under legacy consoles; errors preserve the next request.
"""
import json
import os
import sys
import traceback
from common.engine import InferenceEngine
from common.runtime import configure_runtime

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def event(kind, **data):
    """Emit one UTF-8 JSON event for the desktop worker protocol."""
    print('VOICE_EVENT ' + json.dumps({'kind': kind, **data}, ensure_ascii=False), flush=True)


def main():
    """Run the entry point and manage its application or worker lifecycle."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    configure_runtime()
    engine = InferenceEngine()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            event('done', **engine.generate(request, event))
        except Exception as exc:
            traceback.print_exc()
            event('error', message=engine.describe_error(exc))


if __name__ == '__main__':
    main()
