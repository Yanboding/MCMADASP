"""Writers for the ``run.py`` job-array command (.dat) files.

Two output formats are supported and must be preserved because ``run.py``
parses ``--params`` differently for each:

* :func:`write_grouped_command_file` -- each command's ``--params`` payload is a
  JSON *array* of records (used by the test-path generator).
* :func:`write_command_file` -- each command's ``--params`` payload is a single
  JSON *object* (used by the training / policy-efficiency generators).
"""

import json


def _command_line(index, payload):
    return f"{index} python run.py --params '" + json.dumps(payload) + "'\n"


def _write_lines(lines, dat_file):
    if dat_file:
        with open(dat_file, 'w') as f:
            f.writelines(lines)
        print(f"Saved {len(lines)} commands to {dat_file}")
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
