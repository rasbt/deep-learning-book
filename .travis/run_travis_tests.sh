#!/usr/bin/env bash

set -e

source activate testenv

conda install -q -y jupyter matplotlib
pip install watermark
pip install nbformat

nosetests ann

if [[ "$NOTEBOOKS" == "true" ]]; then
    nosetests code
fi
