#!/bin/bash

if [ $# != 2 ] && [ $# != 3 ]; then
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash run_standalone_train_gpu.sh DEVICE_ID MINDRECORD_DIR LOAD_CHECKPOINT_PATH(optional)"
  echo "for example: bash run_standalone_train_gpu.sh 0 /path/mindrecord_dataset /path/load_ckpt"
  echo "if no ckpt, just run: bash run_standalone_train_gpu.sh 0 /path/mindrecord_dataset"
  echo "=============================================================================================================="
exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DEVICE_ID=$1
MINDRECORD_DIR=$2
if [ $# == 3 ];
then
    LOAD_CHECKPOINT_PATH=$3
else
    LOAD_CHECKPOINT_PATH=""
fi

mkdir -p ms_log 
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
export DEVICE_ID=$DEVICE_ID

CONFIG=$(get_real_path "$PROJECT_DIR/../centernetdet_gpu_config.yaml")

python ${PROJECT_DIR}/../train.py  \
    --config_path $CONFIG \
    --distribute=false \
    --need_profiler=false \
    --device_id=$DEVICE_ID \
    --load_checkpoint_path=$LOAD_CHECKPOINT_PATH \
    --save_checkpoint_num=1 \
    --mindrecord_dir=$MINDRECORD_DIR \
    --mindrecord_prefix="s2mc2.train.mind" \
    --visual_image=false \
    --save_result_dir="" > training_log.txt 2>&1 &
