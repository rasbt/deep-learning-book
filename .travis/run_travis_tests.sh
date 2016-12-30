#!/usr/bin/env bash

set -e

nosetests ann

if [[ "$NOTEBOOKS" == "true" ]]; then
    nosetests code
fi
