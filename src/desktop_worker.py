"""JSON-lines worker protocol. The desktop owns this entire process tree."""
import json
import os
import sys
import traceback
from voice_clone.inference import InferenceEngine

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def event(kind, **data):
    print('VOICE_EVENT ' + json.dumps({'kind': kind, **data}, ensure_ascii=False), flush=True)


def main():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
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
