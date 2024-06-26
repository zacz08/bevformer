#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
# PORT=${PORT:-29503}
PORT=${PORT:-28596}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
