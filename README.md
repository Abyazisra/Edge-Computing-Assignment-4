# Edge Computing Assignment 4 (i222056)

Part 1: knowledge distillation from a ResNet-50 teacher (CIFAR-100) into MobileNetV2, SqueezeNet and EfficientNet-Lite0.
Part 2: hardware aware neural architecture search with Optuna on CIFAR-10, followed by ONNX export and CPU benchmarking.

## Setup

```
pip install -r requirements.txt
```

Python 3.9 or newer. A CUDA GPU is strongly recommended for training (Google Colab T4 is enough). The datasets are downloaded by torchvision into `data/` the first time they are needed. Latency numbers are always measured on the CPU.

## Folder layout

```
src/            all scripts
tools/          make_report.py (builds the PDF from results/)
checkpoints/    .pt files written by the training scripts
onnx_models/    exported .onnx files
results/        json / csv logs, figures, screenshots/
run_all.sh      runs everything in order
```

## Step by step

Run from inside `src/`.

1. Teacher (deliverable 1)
   `python train_teacher.py --epochs 40`
   writes `checkpoints/teacher_resnet50.pt`

2. Baseline students without distillation (deliverables 2, 3)
   `python train_baseline.py --student all --epochs 40`
   writes `checkpoints/<student>_baseline.pt`

3. Distilled students (deliverable 4)
   `python train_kd.py --student all --epochs 40 --T 4 --alpha 0.7 --beta 1.0`
   writes `checkpoints/<student>_kd.pt`. `--beta 0` turns the feature loss off.

4. Architecture search (deliverables 5, 6, 7)
   `python nas_search.py --trials 40 --epochs 3 --max_latency 10 --max_size 1.5`
   writes `results/nas_history.csv`, `results/nas_history.json`, `results/best_arch.json`.
   Choose the limits to suit the machine you run on, latency is in ms for one image on CPU and size is in MB.

5. Train the best architecture from scratch (deliverable 8)
   `python nas_final_train.py --epochs 60`
   writes `checkpoints/nas_best.pt`

6. ONNX export (deliverable 10)
   `python to_onnx.py`
   writes `onnx_models/*.onnx` and prints the maximum difference to PyTorch.

7. Inference (deliverable 9)
   ```
   python infer.py --model mobilenetv2 --variant kd --image path/to/image.png
   python infer.py --model nas --image path/to/image.png
   python infer.py --model all --variant both --evaluate
   python onnx_infer.py --model all --variant both --evaluate
   ```
   `--model` is one of mobilenetv2, squeezenet, efficientnet_lite0, nas, all.

8. Benchmark (deliverable 11)
   `python benchmark.py`
   writes `results/benchmark.csv` (size, accuracy, PyTorch and ONNX Runtime latency, peak memory).

9. Report (deliverable 12)
   Put screenshots in `results/screenshots/` then run `python ../tools/make_report.py`. It creates `Report_i222056.pdf` in the repo root.

Every training and inference script accepts `--smoke`, which uses a few random images instead of CIFAR so the whole chain can be tested in a couple of minutes.

## Notes

- The ResNet-50 and students use a stride 1 stem because the images are only 32x32.
- Peak memory is the growth of process RSS while the model runs, measured with psutil.
- NAS candidates that go over the latency or size limit are logged as `rejected` and are never trained.
