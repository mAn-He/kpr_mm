#!/bin/bash

# 작업 시작 로그 출력
echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"


cd . || exit


# conda 환경 활성화
source ~/.bashrc
conda activate caption

# CUDA 11.0 환경 구성
ml purge
ml load cuda/11.0

# Python 스크립트 실행
# python /scratch/ghtmd9277/keypoint_promptable_reidentification/accel_aug_occuld_duke_final.py
python caption/multiturn_caption.py
# 작업 종료 로그 출력
echo "###"
echo "### END DATE=$(date)"
