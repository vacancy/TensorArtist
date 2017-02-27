#/bin/bash -E

TAROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/../ && pwd )"

export PYTHONPATH=$TAROOT:$PYTHONPATH

if [[ $1 == *.py ]]; then
    exec python3 $@ && exit
fi

exec $@

