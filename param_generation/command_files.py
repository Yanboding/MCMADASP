"""Writers for the ``run.py`` job-array command (.dat) files.

Two output formats are supported and must be preserved because ``run.py``
parses ``--params`` differently for each:

* :func:`write_grouped_command_file` -- each command's ``--params`` payload is a
  JSON *array* of records (used by the test-path generator).
* :func:`write_command_file` -- each command's ``--params`` payload is a single
  JSON *object* (used by the training / policy-efficiency generators).
"""

import json
import os
import re

_LINE_RE = re.compile(r"^\d+ (.*)$")


def _command_line(index, payload):
    return f"{index} python run.py --params '" + json.dumps(payload) + "'\n"


def _strip_index(line):
    """The command text of a dat line without its leading line id."""
    match = _LINE_RE.match(line.rstrip('\n'))
    return match.group(1) if match else None


def _write_lines(lines, dat_file):
    """Write ``lines`` to ``dat_file``, APPENDING to an existing file.

    Successive ``generate_params.py`` invocations with the same ``--dat`` thus
    accumulate into one job-array file: existing commands are kept, new ones
    are added after them, commands already present (identical command text)
    are skipped, and line ids are renumbered 1..n so the file stays a valid
    contiguous job array. Delete the file to start over.
    """
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
    """Split ``results`` into ``num_groups`` strided groups, one command each.

    When ``num_groups`` is falsy, each result becomes its own group. Every
    command's ``--params`` payload is the JSON-encoded list of its group.
    ``start_index`` sets the first command's line id, so several blocks of
    groups can be concatenated into one dat file with continuous ids.
    """
    n = num_groups if num_groups and num_groups > 0 else len(results)
    groups = [results[i::n] for i in range(min(n, len(results)))]
    lines = [_command_line(line_index, group) for line_index, group in enumerate(groups, start=start_index)]
    return _write_lines(lines, dat_file)


def write_command_file(results, dat_file=None):
    """Emit one command per result; each ``--params`` payload is one JSON object."""
    lines = [_command_line(line_index, result) for line_index, result in enumerate(results, start=1)]
    return _write_lines(lines, dat_file)
