#!/bin/bash
xvfb-run --auto-servernum --server-args='-screen 0 640x480x24' python -u /data/asac/main.py $@