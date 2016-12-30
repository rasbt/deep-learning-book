#!/usr/bin/env bash

set -e

if [[ "$NOTEBOOKS" == "true" ]]; then
    nosetests code
else
    nosetests ann
fi
