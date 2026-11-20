"""Builds the PDF report from whatever is in results/.
Run it again after the experiments finish so the tables and figures carry real numbers.
Screenshots: drop PNG files into results/screenshots/ (file name becomes the caption)."""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

NAME = "Abyaz Israr"
ROLL = "i222056"
SECTION = "DS7-A"
ASSIGNMENT = "Assignment 4"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, f"Report_{ROLL}.pdf")

body = ParagraphStyle("body", fontName="Times-Roman", fontSize=11, leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
h1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=15, leading=19, spaceBefore=12, spaceAfter=8)
h2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=12, leading=16, spaceBefore=8, spaceAfter=4)
cap = ParagraphStyle("cap", fontName="Times-Italic", fontSize=9.5, leading=12, alignment=TA_CENTER, spaceAfter=10)
cell = ParagraphStyle("cell", fontName="Times-Roman", fontSize=9, leading=11)
code = ParagraphStyle("code", fontName="Courier", fontSize=8.5, leading=11, leftIndent=10, spaceAfter=6)


def jload(name):
    p = os.path.join(RES, name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def cload(name):
    p = os.path.join(RES, name)
    if os.path.exists(p):
        with open(p) as f:
            return list(csv.DictReader(f))
    return []


def P(t, s=body):
    return Paragraph(t, s)


def table(rows, widths=None, header=True):
    data = [[Paragraph(str(c), cell) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.5, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("LINEBELOW", (0, 0), (-1, 0), 1.2, colors.black))
    t.setStyle(TableStyle(st))
    return t


def fmt(v, nd=2):
    return "-" if v in (None, "") else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def figure_accuracy(students):
    names = [s["name"] for s in students if s["base"] and s["kd"]]
    if not names:
        return None
    base = [s["base"]["final"]["top1"] for s in students if s["base"] and s["kd"]]
    kd = [s["kd"]["final"]["top1"] for s in students if s["base"] and s["kd"]]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], base, 0.4, color="white", edgecolor="black", hatch="//", label="baseline")
    ax.bar([i + 0.2 for i in x], kd, 0.4, color="black", edgecolor="black", label="distilled")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("top-1 accuracy (%)")
    ax.set_ylim(0, max(base + kd) * 1.3 + 1e-9)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    path = os.path.join(RES, "fig_kd_accuracy.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def figure_nas(hist):
    ok = [r for r in hist if r["status"] == "trained"]
    bad = [r for r in hist if r["status"] == "rejected"]
    if not hist:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    if ok:
        ax.scatter([float(r["latency_ms"]) for r in ok], [float(r["val_acc"]) for r in ok],
                   marker="o", facecolors="none", edgecolors="black", label="trained")
    if bad:
        ax.scatter([float(r["latency_ms"]) for r in bad], [0] * len(bad), marker="x", color="black",
                   label="rejected")
    ax.set_xlabel("CPU latency (ms)")
    ax.set_ylabel("validation accuracy (%)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = os.path.join(RES, "fig_nas_search.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def build():
    teacher = jload("teacher_resnet50.json")
    students = []
    for n in ("mobilenetv2", "squeezenet", "efficientnet_lite0"):
        students.append({"name": n, "base": jload(f"{n}_baseline.json"), "kd": jload(f"{n}_kd.json")})
    hist = cload("nas_history.csv")
    best = jload("best_arch.json")
    final = jload("nas_best.json")
    bench = cload("benchmark.csv")

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2.3 * cm, rightMargin=2.3 * cm,
                            topMargin=2.2 * cm, bottomMargin=2.2 * cm, title=f"{ASSIGNMENT} {ROLL}")
    s = []

    # title page
    s.append(Spacer(1, 7 * cm))
    s.append(P(f"<b>{ASSIGNMENT}</b>", ParagraphStyle("t", fontName="Times-Bold", fontSize=28, alignment=TA_CENTER, leading=34)))
    s.append(Spacer(1, 1.5 * cm))
    for line in (f"Name: {NAME}", f"Roll Number: {ROLL}", f"Section: {SECTION}"):
        s.append(P(line, ParagraphStyle("d", fontName="Times-Roman", fontSize=14, alignment=TA_CENTER, leading=22)))
    s.append(PageBreak())

    # 1
    s.append(P("1. Overview", h1))
    s.append(P("This report describes my implementation of the two parts of the assignment. Part 1 is a "
               "knowledge distillation pipeline on CIFAR-100 where a ResNet-50 teacher is used to train three "
               "small student networks. Part 2 is a hardware aware neural architecture search on CIFAR-10 "
               "that looks for a small CNN which fits a CPU latency and storage budget. Everything is written "
               "in PyTorch, models are exported to ONNX and run on CPU with ONNX Runtime."))
    s.append(P("All code lives in the <font name='Courier'>src</font> folder. Each script can be run on its own "
               "and writes its checkpoints to <font name='Courier'>checkpoints</font> and its numbers to "
               "<font name='Courier'>results</font>."))

    # 2
    s.append(P("2. Question 1: Knowledge distillation", h1))
    s.append(P("2.1 Teacher", h2))
    s.append(P("The teacher is the torchvision ResNet-50 with two changes for 32x32 images: the first layer is a "
               "3x3 convolution with stride 1 and the max pool is removed. Without this the feature maps "
               "shrink to 1x1 too early. The network is trained from scratch with SGD (momentum 0.9, Nesterov), "
               "cosine learning rate with one warm up epoch, weight decay 5e-4, random crop and horizontal flip. "
               "<font name='Courier'>train_teacher.py</font> saves the checkpoint."))
    s.append(P("2.2 Students", h2))
    s.append(P("The three students are written by hand in <font name='Courier'>models_student.py</font> and adapted to "
               "small images by using stride 1 in the stem. MobileNetV2 and EfficientNet-Lite0 are stacks of "
               "inverted residual blocks; Lite0 uses 5x5 kernels in some stages, ReLU6 and no squeeze and "
               "excitation, as in the original Lite design. SqueezeNet is built from Fire modules with three "
               "max pooling layers and a 1x1 convolution classifier. Every student returns its last feature "
               "map on request, which is what the feature loss uses."))
    s.append(P("2.3 Distillation framework", h2))
    s.append(P("The loss in <font name='Courier'>kd_losses.py</font> is"))
    s.append(P("L = alpha * T^2 * KL(softmax(z_t / T) || softmax(z_s / T)) + (1 - alpha) * CE(z_s, y) + beta * MSE(R(f_s), f_t)", code))
    s.append(P("The first term is soft target distillation with temperature T, the second is the usual hard label "
               "loss, and the third is a feature hint. R is a 1x1 convolution with batch norm that maps the "
               "student feature channels to the 2048 channels of the teacher final stage. If the spatial sizes "
               "differ the student map is average pooled to the teacher size. The regressor is only needed "
               "during training and is not saved with the student. I used T = 4, alpha = 0.7 and beta = 1 "
               "by default. Setting beta to 0 gives plain logit distillation."))
    s.append(P("2.4 Baselines, inference and ONNX", h2))
    s.append(P("<font name='Courier'>train_baseline.py</font> trains the same three students with identical "
               "optimizer settings and epochs but with the cross entropy loss only, so the comparison is fair. "
               "<font name='Courier'>infer.py</font> loads any trained model (students in both variants and the "
               "NAS model) and either classifies an image or evaluates the test split. "
               "<font name='Courier'>to_onnx.py</font> exports each model with a dynamic batch axis, checks "
               "it with the ONNX checker and compares the output with PyTorch. "
               "<font name='Courier'>onnx_infer.py</font> runs the exported files with the CPU execution provider."))

    # 3
    s.append(P("3. Question 2: Neural architecture search", h1))
    s.append(P("3.1 Search space and generator", h2))
    s.append(P("A candidate is described by a small dictionary: input resolution (24, 32 or 40), activation "
               "(ReLU, ReLU6, SiLU, Hardswish or LeakyReLU), stem width, and between 2 and 4 stages. Every stage has "
               "a depth from 1 to 3 blocks, a width from 16 to 128, a kernel size of 3, 5 or 7, and a block type, "
               "either a normal convolution or a depthwise separable block. Stages after the first downsample "
               "with stride 2. <font name='Courier'>nas_space.py</font> turns such a dictionary into a "
               "PyTorch model. The resize to the searched resolution is done inside the model so every "
               "candidate takes the same 32x32 input."))
    s.append(P("3.2 Search strategy", h2))
    s.append(P("I use Optuna with the TPE sampler. The parameters are sampled in a define by run style so the "
               "number of stages decides which stage parameters exist. For each trial the model is built, then "
               "CPU latency (median of 100 single image runs after warm up), saved weight size and peak memory growth "
               "during inference are measured. If latency or size is above the limit the candidate is "
               "rejected and never trained. Otherwise it is trained for a few epochs on a subset of CIFAR-10 and "
               "scored on a held out validation set. The score is validation accuracy minus a small penalty "
               "proportional to latency. Each trial is appended to "
               "<font name='Courier'>results/nas_history.csv</font> and the best one is stored in "
               "<font name='Courier'>results/best_arch.json</font>."))
    s.append(P("3.3 Final training and deployment", h2))
    s.append(P("<font name='Courier'>nas_final_train.py</font> rebuilds the best configuration, trains it from "
               "scratch on the full training set for more epochs and reports test accuracy. The model is exported to "
               "ONNX and used through the same inference and benchmark scripts as the students."))

    # 4
    s.append(PageBreak())
    s.append(P("4. Results", h1))
    s.append(P("4.1 Teacher and students on CIFAR-100", h2))
    rows = [["Model", "Params (M)", "Epochs", "Top-1 (%)", "Top-5 (%)"]]
    if teacher:
        rows.append(["ResNet-50 teacher", "23.7", teacher["epochs"], fmt(teacher["final"]["top1"]), fmt(teacher["final"]["top5"])])
    for st in students:
        for key, lab in (("base", "baseline"), ("kd", "distilled")):
            r = st[key]
            if r:
                rows.append([f"{st['name']} ({lab})", fmt(r["params"] / 1e6, 2), r["epochs"],
                             fmt(r["final"]["top1"]), fmt(r["final"]["top5"])])
    if len(rows) == 1:
        rows.append(["no results yet", "-", "-", "-", "-"])
    s.append(table(rows))
    s.append(Spacer(1, 6))
    fig = figure_accuracy(students)
    if fig:
        s.append(Image(fig, width=11 * cm, height=6.3 * cm))
        s.append(P("Figure 1. Test accuracy of each student with and without distillation.", cap))
        gains = []
        for st in students:
            if st["base"] and st["kd"]:
                d = st["kd"]["final"]["top1"] - st["base"]["final"]["top1"]
                gains.append(f"{st['name']} {d:+.2f} points")
        s.append(P("Change in top-1 accuracy after distillation: " + ", ".join(gains) + "."))

    s.append(P("4.2 Architecture search on CIFAR-10", h2))
    if hist and best:
        n_ok = sum(r["status"] == "trained" for r in hist)
        s.append(P(f"The search evaluated {len(hist)} candidates. {len(hist) - n_ok} were rejected before training for "
                   f"breaking the constraints (latency at most {best['constraints']['max_latency_ms']} ms, "
                   f"size at most {best['constraints']['max_size_mb']} MB)."))
        figs = figure_nas(hist)
        if figs:
            s.append(Image(figs, width=11 * cm, height=6.3 * cm))
            s.append(P("Figure 2. Candidates seen by the search. Rejected ones are drawn at zero accuracy.", cap))
        cfg = best["config"]
        rows = [["Setting", "Value"], ["Input resolution", cfg["resolution"]], ["Activation", cfg["act"]],
                ["Stem width", cfg["stem"]]]
        for i, st in enumerate(cfg["stages"]):
            rows.append([f"Stage {i + 1}", f"{st['depth']} x {st['block']}, {st['width']} ch, kernel {st['kernel']}"])
        m = best["metrics"]
        rows += [["Search val. accuracy (%)", fmt(m["val_acc"])], ["CPU latency (ms)", fmt(m["latency_ms"], 3)],
                 ["Weights size (MB)", fmt(m["size_mb"], 3)], ["Parameters", m["params"]]]
        if final:
            rows.append(["Final test accuracy (%)", fmt(final["final"]["top1"])])
            rows.append(["Final training epochs", final["epochs"]])
        s.append(table(rows, widths=[6 * cm, 9 * cm]))
    else:
        s.append(P("No search results found in the results folder."))

    s.append(P("4.3 CPU benchmark", h2))
    rows = [["Model", "Variant", "Params (M)", "ONNX (MB)", "Top-1 ONNX", "PyTorch (ms)", "ORT (ms)", "Peak mem (MB)"]]
    for r in bench:
        rows.append([r["model"], r["variant"], r["params_M"], r["onnx_size_MB"], r["top1_onnx"] or "-",
                     r["torch_lat_ms"], r["ort_lat_ms"], r["peak_mem_MB"]])
    if len(rows) == 1:
        rows.append(["no results yet"] + ["-"] * 7)
    s.append(table(rows))
    s.append(P("Latency is the median of single image runs on CPU with one thread. Peak memory is the growth of the "
               "process resident memory while a model is loaded and run.", cap))

    # 5
    shots = sorted(glob.glob(os.path.join(RES, "screenshots", "*.png")))
    s.append(PageBreak())
    s.append(P("5. Screenshots", h1))
    if shots:
        for p in shots:
            from PIL import Image as PILImage
            w, h = PILImage.open(p).size
            width = min(15.5 * cm, w)
            s.append(Image(p, width=width, height=width * h / w))
            s.append(P(os.path.splitext(os.path.basename(p))[0].replace("_", " "), cap))
    else:
        s.append(P("Screenshots of the training runs, NAS search, ONNX export and benchmark go here."))

    # 6
    s.append(P("6. How to run", h1))
    s.append(P("Install the packages from requirements.txt, then run <font name='Courier'>run_all.sh</font>, or "
               "run the scripts in <font name='Courier'>src</font> one by one in this order: train_teacher, "
               "train_baseline, train_kd, nas_search, nas_final_train, to_onnx, infer, onnx_infer, benchmark. "
               "The README lists the options of every script."))
    doc.build(s)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
