#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

PYTHON_EXE="python"
PYTHON_VERSION=$("$PYTHON_EXE" --version 2>&1)
$(echo "$PYTHON_VERSION" | grep -q "3.9")
if [[ "$?" -ne 0 ]]; then
    PYTHON_EXE="python3"
    PYTHON_VERSION=$("$PYTHON_EXE" --version 2>&1)
    $(echo "$PYTHON_VERSION" | grep -q "3.9")
    if [[ "$?" -ne 0 ]]; then
        echo "Error: Using python version $PYTHON_VERSION. Please use Python 3.9"
        exit 1
    fi
fi

cd "$SCRIPT_DIR/.."
$("$PYTHON_EXE" -m venv ./venv)
source "./venv/bin/activate"

WHEEL=$(ls | grep *.whl)
pip install "$WHEEL"