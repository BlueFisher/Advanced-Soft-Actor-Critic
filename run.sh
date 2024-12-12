#!/bin/bash
set -e

/etc/entrypoint.sh
export DISPLAY=:0
export VGL_DISPLAY=egl
export MPLBACKEND=agg
eval "$*"