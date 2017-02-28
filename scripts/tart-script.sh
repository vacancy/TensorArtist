#!/bin/bash -E

TAROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/../ && pwd )"
SCRIPT=$( basename ${BASH_SOURCE[0]} )
SCRIPT=${SCRIPT:5}

export PYTHONPATH=$TAROOT:$PYTHONPATH
exec python3 "$TAROOT/scripts/$SCRIPT.py" $@

