#!/bin/bash

config_path=$1

python3 train.py --cfg-path "$config_path"
