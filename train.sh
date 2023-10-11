#!/bin/bash
set -e

/etc/entrypoint.sh
export DISPLAY=:0
export MPLBACKEND=agg
python -u /data/asac/main.py $@