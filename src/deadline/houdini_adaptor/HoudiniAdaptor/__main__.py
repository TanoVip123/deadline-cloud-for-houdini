# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

import json as _json
import logging as _logging
import os as _os
import sys as _sys
from typing import List as _List
from typing import Mapping as _Mapping

import yaml as _yaml
from openjd.adaptor_runtime import EntryPoint as _EntryPoint

from .adaptor import HoudiniAdaptor

__all__ = ["main"]
_logger = _logging.getLogger(__name__)

# A PDG work item's run data gets these from the task's session. The daemon started
# before the task, so it cannot read them itself.
_PDG_TASK_ID_VARIABLES: _Mapping[str, str] = {
    "farm_id": "DEADLINE_FARM_ID",
    "queue_id": "DEADLINE_QUEUE_ID",
    "job_id": "DEADLINE_JOB_ID",
    "step_id": "DEADLINE_STEP_ID",
    "task_id": "DEADLINE_TASK_ID",
}


def _with_pdg_task_ids(argv: _List[str], env: _Mapping[str, str]) -> _List[str]:
    """
    Returns the arguments with the task's IDs added to the ``pdg`` object of ``--run-data``.
    Run data without a ``pdg`` object, or that cannot be read, is left for the runtime to handle.
    """
    # The run data is either the argument after --run-data or the value after "--run-data=".
    index: int
    prefix: str
    value: str
    for i, arg in enumerate(argv):
        if arg == "--run-data" and i + 1 < len(argv):
            index, prefix, value = i + 1, "", argv[i + 1]
        elif arg.startswith("--run-data="):
            index, prefix, value = i, "--run-data=", arg[len("--run-data=") :]
        else:
            continue
        try:
            run_data: object = _read_run_data(value)
        except (OSError, _yaml.YAMLError):
            return argv
        if not isinstance(run_data, dict) or not isinstance(run_data.get("pdg"), dict):
            return argv
        # Add each ID the session has, and write the run data back inline as JSON.
        for key, variable in _PDG_TASK_ID_VARIABLES.items():
            if env.get(variable):
                run_data["pdg"][key] = env[variable]
        result: _List[str] = list(argv)
        result[index] = prefix + _json.dumps(run_data)
        return result
    return argv


def _read_run_data(value: str) -> object:
    """The run data, read the way the runtime reads it: a file:// path, or inline JSON or YAML."""
    if value.startswith("file://"):
        with open(value[len("file://") :], encoding="utf-8") as f:
            return _yaml.safe_load(f)
    return _yaml.safe_load(value)


def main(reentry_exe=None) -> int:
    """
    Entry point for the Houdini Adaptor
    """
    _logger.info("About to start the HoudiniAdaptor")

    package_name = vars(_sys.modules[__name__])["__package__"]
    if not package_name:
        raise RuntimeError(f"Must be run as a module. Do not run {__file__} directly")

    # The task's IDs reach the daemon only through the run data.
    _sys.argv = _with_pdg_task_ids(_sys.argv, _os.environ)

    try:
        _EntryPoint(HoudiniAdaptor).start(reentry_exe=reentry_exe)
    except Exception as e:
        _logger.error(f"Entrypoint failed: {e}")
        return 1

    _logger.info("Done HoudiniAdaptor main")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
