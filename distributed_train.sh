#!/bin/bash
# NUM_PROC=$1
# LOCALPORT=$2
# shift
# torchrun --nproc_per_node=$NUM_PROC --rdzv-backend=c10d --rdzv-endpoint=localhost:$LOCALPORT train.py "$@"

NUM_PROC=$1
shift
torchrun --nproc_per_node=$NUM_PROC train.py "$@"
