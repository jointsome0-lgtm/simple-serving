#!/usr/bin/env python3
"""Delete the pictures ComfyUI leaves in its temp directory, and the job records nobody collected.

A PreviewImage node writes the picture of a reader's scene to ComfyUI's temp directory before /view hands it to the
bot, and no route of the HTTP API deletes that file: the pinned server empties the directory only when it starts. The
bot deletes the job's /history record the moment it has the picture (local/image-batch.ts `drawOne`), so a temp file
that no record names any more is a picture already delivered or given up. This runs beside the server
(gpu/image-serve.sh), reads /history on loopback once a second and deletes:
  - a temp file no record names, once it is older than the grace. The grace covers the moment between the node
    writing the file and the server writing the job's record, when a picture about to be fetched is named nowhere;
  - any temp file older than the file cap, named or not: a record the bot never deleted must not keep its picture;
  - a record whose job ended longer ago than the history cap: the bot died between the drawing and its delete, and
    the record holds the whole prompt.
With /history unreadable only the file cap applies, because then nothing says which picture is still wanted.
It prints counts and codes only, never a file name or anything read from a record: both are somebody's scene.
Stdlib only, so that it runs under ComfyUI's own virtual environment.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def emit(event, **fields):
    print(json.dumps({'event': event, **fields}, separators=(',', ':')), flush=True)


def alive(pid):
    """Whether the server is still there. A zombie is not: it only waits for its parent to collect it."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f'/proc/{pid}/stat', 'rb') as stat:
            # The state follows the command name, which is in parentheses and may itself hold spaces or a ')'.
            return stat.read().rsplit(b')', 1)[1].split()[0] != b'Z'
    except (OSError, IndexError):
        return True


def failure_code(error):
    """A word from a closed set for a failed request: the error's own text may carry a path or a server's reply."""
    if isinstance(error, urllib.error.HTTPError):
        return f'http_{error.code}' if 100 <= error.code <= 599 else 'http_other'
    reason = getattr(error, 'reason', error)
    if isinstance(reason, ConnectionRefusedError):
        return 'connection_refused'
    if isinstance(reason, TimeoutError):
        return 'timeout'
    if isinstance(error, json.JSONDecodeError) or isinstance(error, UnicodeDecodeError):
        return 'invalid_json'
    if isinstance(error, ValueError):
        return 'invalid_shape'
    if isinstance(error, OSError):
        return 'os_error'
    return 'other'


def read_history(base):
    with urllib.request.urlopen(f'{base}/history', timeout=5) as response:
        value = json.loads(response.read().decode('utf-8'))
    if not isinstance(value, dict):
        raise ValueError('shape')
    return value


def delete_records(base, ids):
    request = urllib.request.Request(f'{base}/history', data=json.dumps({'delete': ids}).encode('utf-8'),
                                     headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=5) as response:
        response.read()


def named_files(history):
    """The temp files the records still name, relative to the temp directory. Every list in a node's output is read,
    not only `images`: whatever a node wrote there is named the same way, by filename, subfolder and type."""
    names = set()
    for record in history.values():
        outputs = record.get('outputs') if isinstance(record, dict) else None
        for output in outputs.values() if isinstance(outputs, dict) else ():
            for items in output.values() if isinstance(output, dict) else ():
                for item in items if isinstance(items, list) else ():
                    if not isinstance(item, dict) or item.get('type') != 'temp' or not isinstance(item.get('filename'), str):
                        continue
                    subfolder = item.get('subfolder') if isinstance(item.get('subfolder'), str) else ''
                    names.add(os.path.normpath(os.path.join(subfolder, item['filename'])))
    return names


def ended_at(record):
    """When the job of a record ended, in seconds, from the server's own message timestamps, or None. The pinned
    server stamps every status message in milliseconds (execution.py `add_message`) on this machine's clock."""
    status = record.get('status') if isinstance(record, dict) else None
    messages = status.get('messages') if isinstance(status, dict) else None
    stamps = []
    for message in messages if isinstance(messages, list) else ():
        data = message[1] if isinstance(message, list) and len(message) == 2 else None
        stamp = data.get('timestamp') if isinstance(data, dict) else None
        if isinstance(stamp, (int, float)) and not isinstance(stamp, bool):
            stamps.append(stamp / 1000)
    return max(stamps) if stamps else None


def stale_records(history, now, cap, first_seen):
    """Ids of the records whose job ended at least `cap` seconds ago. A record without a timestamp is timed from the
    pass that first saw it, so it goes too, one cap later than it would have."""
    for key in [key for key in first_seen if key not in history]:
        del first_seen[key]
    stale = []
    for key, record in history.items():
        ended = ended_at(record)
        if ended is None:
            ended = first_seen.setdefault(key, now)
        if now - ended >= cap:
            stale.append(key)
    return stale


def sweep(temp, history, now, grace, file_cap):
    """Deletes the temp files that are due and the subfolders they leave empty. Returns (deleted, failed)."""
    named = named_files(history) if history is not None else set()
    deleted = failed = 0
    for root, directories, files in os.walk(temp, topdown=False):
        # A link is deleted as the link, whether it points at a file or a directory; the walk never follows one.
        links = [name for name in directories if os.path.islink(os.path.join(root, name))]
        for name in files + links:
            path = os.path.join(root, name)
            try:
                age = now - os.lstat(path).st_mtime
            except FileNotFoundError:
                continue
            relative = os.path.normpath(os.path.relpath(path, temp))
            # Bot portraits are uploaded before a job exists and may wait in its queue. The bot overwrites them
            # with a blank PNG on every exit; a crash or failed overwrite leaves at most this ten-minute cap.
            reference = relative.startswith('bot-references' + os.sep)
            unnamed = history is not None and relative not in named
            if age < file_cap and (reference or not (unnamed and age >= grace)):
                continue
            try:
                os.unlink(path)
                deleted += 1
            except FileNotFoundError:
                pass
            except OSError:
                failed += 1
        # A subfolder the server made for a file it is about to write is as young as that file, so the grace
        # protects it too; rmdir refuses one that is not empty.
        if root != temp:
            try:
                if now - os.lstat(root).st_mtime >= grace:
                    os.rmdir(root)
            except OSError:
                pass
    return deleted, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--pid', type=int, required=True, help='the ComfyUI process; the sweeper stops when it is gone')
    parser.add_argument('--temp', required=True, help="ComfyUI's temp directory itself, with the `temp` the server appends")
    parser.add_argument('--port', type=int, required=True, help="ComfyUI's port on 127.0.0.1")
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('--grace', type=float, default=5.0)
    parser.add_argument('--file-cap', type=float, default=600.0)
    parser.add_argument('--history-cap', type=float, default=600.0)
    parser.add_argument('--once', action='store_true', help='one pass, then exit')
    args = parser.parse_args()
    base = f'http://127.0.0.1:{args.port}'
    temp = os.path.abspath(args.temp)
    first_seen = {}
    # The code of the last failed read: a failure is printed when it starts or changes, not once a second.
    unreadable = None
    emit('sweeper_started', graceSeconds=args.grace, fileCapSeconds=args.file_cap, historyCapSeconds=args.history_cap)
    while True:
        if not alive(args.pid):
            emit('sweeper_stopped', reason='server_gone')
            return 0
        now = time.time()
        history = None
        try:
            history = read_history(base)
            if unreadable is not None:
                emit('history_readable')
                unreadable = None
        except Exception as error:
            code = failure_code(error)
            if code != unreadable:
                emit('history_unreadable', code=code)
                unreadable = code
        records = 0
        if history is not None:
            stale = stale_records(history, now, args.history_cap, first_seen)
            if stale:
                try:
                    delete_records(base, stale)
                    records = len(stale)
                    # Their files are named by nothing now, and go in this same pass once past the grace.
                    for key in stale:
                        history.pop(key, None)
                except Exception as error:
                    emit('records_undeleted', code=failure_code(error), count=len(stale))
        try:
            deleted, failed = sweep(temp, history, now, args.grace, args.file_cap)
        except Exception as error:
            emit('sweep_failed', code=failure_code(error))
        else:
            if deleted or failed or records:
                emit('swept', files=deleted, failedFiles=failed, records=records)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
