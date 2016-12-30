#!/usr/bin/env bash

set -e

source activate testenv

nosetests ann

if [[ "$NOTEBOOKS" == "true" ]]; then
    nosetests code
fi
