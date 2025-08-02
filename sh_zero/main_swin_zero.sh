#!/bin/bash

# 작업 시작 로그 출력
echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"


cd /scratch/ghtmd9277/keypoint_promptable_reidentification || exit


# conda 환경 활성화
source ~/.bashrc
conda activate hs_lsma

# CUDA 11.0 환경 구성
ml purge
ml load cuda/11.0

# Python 스크립트 실행
python /scratch/ghtmd9277/keypoint_promptable_reidentification/main.py --config-file /scratch/ghtmd9277/keypoint_promptable_reidentification/configs/kpr/imagenet/kpr_market_test_zeroshot_text.yaml
# 작업 종료 로그 출력
echo "###"
echo "### END DATE=$(date)"
