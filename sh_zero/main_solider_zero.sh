#!/bin/bash

# 작업 시작 로그 출력
echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"


cd . || exit


# conda 환경 활성화
source ~/.bashrc
conda activate hs_lsma

# CUDA 11.0 환경 구성
ml purge
ml load cuda/11.0

# Python 스크립트 실행
python main.py --config-file configs/kpr/solider/kpr_market_zeroshot_text.yaml

# 작업 종료 로그 출력
echo "###"
echo "### END DATE=$(date)"
