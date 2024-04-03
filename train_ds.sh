#!/bin/bash
set -e

/etc/entrypoint.sh
export DISPLAY=:0
export VGL_DISPLAY=egl
export MPLBACKEND=agg
python -u /data/asac/main_ds.py $@