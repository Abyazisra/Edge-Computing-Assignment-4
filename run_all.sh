#!/usr/bin/env bash
# Runs the whole assignment from the repo root. Needs a GPU for sensible training times.
# Add --smoke to every python call for a quick dry run on random data.
set -e
cd "$(dirname "$0")/src"

# Q1
python train_teacher.py  --epochs 40
python train_baseline.py --student all --epochs 40
python train_kd.py       --student all --epochs 40 --T 4 --alpha 0.7 --beta 1.0

# Q2
python nas_search.py     --trials 40 --epochs 3 --max_latency 10 --max_size 1.5
python nas_final_train.py --epochs 60

# deployment and benchmarking
python to_onnx.py
python infer.py      --model all --variant both --evaluate
python onnx_infer.py --model all --variant both --evaluate
python benchmark.py

# report with the real numbers
python ../tools/make_report.py
