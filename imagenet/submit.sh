#!/bin/bash
source ~/.functions.sh
gpucentos

ls -l /data01

scp -r rennsrdgpu01:/data01/IN /data01

make train
