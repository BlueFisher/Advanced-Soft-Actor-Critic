#!/bin/bash
set -e

/etc/bootstrap.sh
export DISPLAY=:0
export MPLBACKEND=agg
python -u /data/asac/main_oc.py $@