#!/bin/bash

/etc/bootstrap.sh
export DISPLAY=:0
python -u /data/asac/main_ds.py $@