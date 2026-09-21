#!/usr/bin/env bash
# Run the test suite in the project venv.
#
# This machine has ROS 2 sourced into the shell, which injects /opt/ros onto
# PYTHONPATH and makes pytest try to autoload ROS's launch_testing plugin
# (which then fails on a missing 'lark'). We clear PYTHONPATH and disable
# third-party plugin autoloading so tests run against the venv only.
set -euo pipefail
cd "$(dirname "$0")/.."
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest "$@"
