#!/usr/bin/env bash

set -e

nosetests ann

if [[ "$NOTEBOOKS" == "true" ]]; then
    find code -name "*.ipynb" -exec jupyter nbconvert --to notebook --execute {} \;
fi
