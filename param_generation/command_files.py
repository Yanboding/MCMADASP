import json
import os
import re

_LINE_RE = re.compile(r"^\d+ (.*)$")


def _command_line(index, payload):
    return f"{index} python run.py --params '" + json.dumps(payload) + "'\n"


def _strip_index(line):
    match = _LINE_RE.match(line.rstrip('\n'))
    return match.group(1) if match else None


def _write_lines(lines, dat_file):
    if not dat_file:
        return lines
    existing = []
    if os.path.exists(dat_file):
        with open(dat_file) as f:
            existing = [_strip_index(line) for line in f if line.strip()]
        existing = [command for command in existing if command is not None]
    seen = set(existing)
    added = []
    for line in lines:
        command = _strip_index(line)
        if command not in seen:
            seen.add(command)
            added.append(command)
    commands = existing + added
    with open(dat_file, 'w') as f:
        f.writelines(f"{index} {command}\n" for index, command in enumerate(commands, start=1))
    if existing:
        print(f"Appended {len(added)} new commands to {dat_file} "
              f"({len(lines) - len(added)} already present, {len(commands)} total)")
    else:
        print(f"Saved {len(commands)} commands to {dat_file}")
    return lines


def write_grouped_command_file(results, num_groups=None, dat_file=None, start_index=1):
    n = num_groups if num_groups and num_groups > 0 else len(results)
    groups = [results[i::n] for i in range(min(n, len(results)))]
    lines = [_command_line(line_index, group) for line_index, group in enumerate(groups, start=start_index)]
    return _write_lines(lines, dat_file)


def write_command_file(results, dat_file=None):
    lines = [_command_line(line_index, result) for line_index, result in enumerate(results, start=1)]
    return _write_lines(lines, dat_file)
