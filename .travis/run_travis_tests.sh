#!/usr/bin/env bash

set -e

if [[ "$NOTEBOOKS" == "true" ]]; then
    nosetests -s -v code
else
    nosetests -s -v ann
fi
