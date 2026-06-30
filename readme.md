<div align="center">

<h1>
  <span style="font-size:64px; font-weight:800; line-height:1.15; background:linear-gradient(90deg,#46cbc7,#f8fafc 50%,#de5a1d); -webkit-background-clip:text; background-clip:text; color:transparent;">
    InstanceControl
  </span>
</h1>

### Controllable Complex Image Generation without Instance Labeling

### Accepted by ECCV 2026

<p>
  <a href="https://eccv.ecva.net/">
    <img src="https://img.shields.io/badge/ECCV-2026-46cbc7" alt="ECCV 2026">
  </a>
  <a href="https://instancecontrol.github.io/InstanceControl/">
    <img src="https://img.shields.io/badge/Homepage-Project-46cbc7" alt="Project Page">
  </a>
  <a href="#citation">
    <img src="https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://huggingface.co/xiaoyu1104">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow" alt="Hugging Face Models">
  </a>
  <a href="https://huggingface.co/datasets/xiaoyu1104/MIG_train">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-MIG--Train-orange" alt="MIG-Train">
  </a>
  <a href="https://huggingface.co/datasets/xiaoyu1104/MIG-Eval">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark-MIG--Eval-de5a1d" alt="MIG-Eval">
  </a>
</p>

<p>
  <b>InstanceControl</b> is a multi-instance controllable generation method without instance labeling. It automatically associates text prompts with visual conditions at the instance level, enabling high-fidelity generation.
</p>

</div>

---

## 🔥 News

- 🎉 InstanceControl has been accepted by ECCV 2026.
- ✅ Pretrained InstanceControl checkpoints for Canny, depth, and HED have been released on Hugging Face.
- ✅ MIG-Train and MIG-Eval are available on Hugging Face.
- ✅ Inference, training, and evaluation code are now included in this repository.
- More updates are coming soon. Stay tuned and ⭐ star the repo!

## TODO

- [x] Release pretrained checkpoints.
- [x] Release MIG-Train and MIG-Eval.
- [x] Release inference pipeline.
- [x] Release training and evaluation code.

## Installation

```bash
conda create -n instancecontrol python=3.10
conda activate instancecontrol

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Model Zoo

📦 Download our checkpoints and place them under `pretrain_model/`.

| Condition | InstanceControl Checkpoint | Stage 1 | Stage 2 |
| --- | --- | --- | --- |
| Canny | [🤗 InstanceControl_canny](https://huggingface.co/xiaoyu1104/InstanceControl_canny) | `Sa2va-Instance-4B/` | `FLUX-Control-Canny/` |
| Depth | [🤗 InstanceControl_depth](https://huggingface.co/xiaoyu1104/InstanceControl_depth) | `Sa2va-Instance-4B/` | `FLUX-Control-Depth/` |
| HED | [🤗 InstanceControl_hed](https://huggingface.co/xiaoyu1104/InstanceControl_hed) | `Sa2va-Instance-4B/` | `FLUX-Control-Hed/` |

📦 Also download the official [🤗 FLUX.1-Canny-dev](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev) and [🤗 FLUX.1-Depth-dev](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev) models, and place them under `pretrain_model/`.

Expected checkpoint layout:

```text
pretrain_model/
├── InstanceControl_canny/
│   ├── Sa2va-Instance-4B/
│   └── FLUX-Control-Canny/
├── InstanceControl_depth/
│   ├── Sa2va-Instance-4B/
│   └── FLUX-Control-Depth/
├── InstanceControl_hed/
│   ├── Sa2va-Instance-4B/
│   └── FLUX-Control-Hed/
├── FLUX.1-Canny-dev/
└── FLUX.1-Depth-dev/
```

## Quick Start

The repository includes example condition images and prompt JSON files under `example/`.

### Step 1: Predict Instance Masks

<details open>
<summary><b>Canny</b></summary>

```bash
python stage1_Sa2VA/projects/llava_sam2/evaluation/gcg_eval_our_folders.py \
  --model_path ./pretrain_model/InstanceControl_canny/Sa2va-Instance-4B \
  --image_dir ./example/canny \
  --json_dir ./example/json \
  --save_dir ./results/json_pred_canny
```

</details>

<details>
<summary><b>Depth</b></summary>

```bash
python stage1_Sa2VA/projects/llava_sam2/evaluation/gcg_eval_our_folders.py \
  --model_path ./pretrain_model/InstanceControl_depth/Sa2va-Instance-4B \
  --image_dir ./example/depth \
  --json_dir ./example/json \
  --save_dir ./results/json_pred_depth
```

</details>

<details>
<summary><b>HED</b></summary>

```bash
python stage1_Sa2VA/projects/llava_sam2/evaluation/gcg_eval_our_folders.py \
  --model_path ./pretrain_model/InstanceControl_hed/Sa2va-Instance-4B \
  --image_dir ./example/hed \
  --json_dir ./example/json \
  --save_dir ./results/json_pred_hed
```

</details>

Optional mask visualization:

```bash
python stage1_Sa2VA/projects/llava_sam2/evaluation/visualize_mask2.py \
  --image_dir ./example/canny \
  --json_dir ./results/json_pred_canny \
  --save_dir ./results/json_pred_canny_vis
```

### Step 2: Generate Images

<details open>
<summary><b>Canny generation</b></summary>

```bash
python stage2_FLUX_ControlNet/test_predict.py \
  --model_path ./FLUX.1-Canny-dev \
  --checkpoint_dir ./pretrain_model/InstanceControl_canny/FLUX-Control-Canny \
  --canny_pth ./example/canny \
  --box_pth ./results/json_pred_canny \
  --result_path ./results/generation_image_canny 
```

</details>

<details>
<summary><b>Depth generation</b></summary>

```bash
python stage2_FLUX_ControlNet/test_predict.py \
  --model_path ./FLUX.1-Depth-dev \
  --checkpoint_dir ./pretrain_model/InstanceControl_depth/FLUX-Control-Depth \
  --canny_pth ./example/depth \
  --box_pth ./results/json_pred_depth \
  --result_path ./results/generation_image_depth 
```

</details>


<details>
<summary><b>Hed generation</b></summary>

```bash
python stage2_FLUX_ControlNet/test_predict.py \
  --model_path ./FLUX.1-Canny-dev \
  --checkpoint_dir ./pretrain_model/InstanceControl_hed/FLUX-Control-Hed \
  --canny_pth ./example/hed \
  --box_pth ./results/json_pred_hed \
  --result_path ./results/generation_image_hed 
```

</details>

## Training

### Step 1: Data Preparation

Training data is built from images, captions, instance masks, and visual condition maps. We use images from the [📦 Segment Anything Dataset](https://ai.meta.com/datasets/segment-anything-downloads/), [🖼️ COCO 2017](https://cocodataset.org/#download), and [🤗 UniWorld-V1 / BLIP3o-60k](https://huggingface.co/datasets/LanguageBind/UniWorld-V1/tree/main/data/BLIP3o-60k).

⚙️ After organizing the raw images, generate the corresponding Canny, HED, and depth condition maps with `bash dataset/train/data_pro/get_visual_condition/get_condition.sh`.

The prepared training annotations can be downloaded from [🤗 MIG_train](https://huggingface.co/datasets/xiaoyu1104/MIG_train). If you want to build annotations for a custom dataset, please follow [📘 this README](dataset/train/data_pro/get_caption/readme.md) to generate the required JSON files.

Expected data layout:

```text
data/
├── gene/
│   ├── dalle3/
│   ├── geneval_train/
│   ├── JourneyDB/
│   ├── MSCOCO_human/
│   ├── object_2/
│   ├── occupation_1/
│   └── occupation_2/
├── sam/
└── coco/
```

Each dataset split should contain:

```text
sam/
├── image/
├── json/
├── canny/
├── hed/
├── depth/
└── masks/
```

### Step 2: Training Environment

The training environment builds on the inference environment. Install the additional training dependencies before launching the training scripts:

```bash
pip install -r requirements_train.txt
pip install -U openmim
mim install mmcv==2.2.0
```

### Step 3: Train Stage 1

Stage 1 trains the instance parsing model that predicts phrase-level masks from the input condition image and caption. Before training, download [🤗 sam2_hiera_large.pt](https://huggingface.co/facebook/sam2-hiera-large), [🤗 InternVL2_5-4B](https://huggingface.co/OpenGVLab/InternVL2_5-4B), and [🤗 Sa2VA-4B](https://huggingface.co/ByteDance/Sa2VA-4B), then place them under `stage1_Sa2VA/pretrained/`.

```text
stage1_Sa2VA/
└── pretrained/
    ├── sam2_hiera_large.pt
    ├── InternVL2_5-4B/
    └── Sa2VA-4B/
```

⚙️ Generate the Stage 1 training JSON after the data and pretrained models are prepared:

```bash
python dataset/train/data_pro/gene_json_stage1.py
```

🚀 Start Stage 1 training from the `stage1_Sa2VA` directory:

```bash
cd stage1_Sa2VA
bash train.sh
```

#### Convert the Trained Model to Hugging Face Format

After Stage 1 training, convert the trained checkpoint into Hugging Face format with the script below. Replace `PATH_TO_PTH_MODEL` with the trained `.pth` checkpoint and `PATH_TO_SAVE_FOLDER` with the folder where the converted model should be saved.

```bash
python convert_to_hf.py projects/llava_sam2/configs/sa2va_4b.py \
  --pth-model PATH_TO_PTH_MODEL \
  --save-path PATH_TO_SAVE_FOLDER
```

After conversion, replace the generated `sam2.py` and `modeling_sa2va_chat.py` in `PATH_TO_SAVE_FOLDER` with the customized versions from `stage1_Sa2VA/sam2.py` and `stage1_Sa2VA/modeling_sa2va_chat.py`.

### Step 4: Train Stage 2 with GT Masks

This stage trains the FLUX ControlNet branch using ground-truth instance masks. Download the corresponding FLUX ControlNet base model, such as [🤗 FLUX.1-Canny-dev](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev) or [🤗 FLUX.1-Depth-dev](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev), and place it under `stage2_FLUX_ControlNet_GT/pretrained/`.

⚙️ Generate the Stage 2 JSON file for GT-mask training:

```bash
python dataset/train/data_pro/gene_json_stage2_gt.py
```

🚀 Start GT-mask training:

```bash
cd stage2_FLUX_ControlNet_GT
bash train.sh
```

### Step 5: Train Stage 2 with Predicted Masks

This stage trains with masks predicted by Stage 1. First run Stage 1 on the training set, then merge the predictions with ground-truth metadata and convert them into the JSON format required by Stage 2:

```bash
python dataset/train/data_pro/add_gt_mask.py
python dataset/train/data_pro/change_premask_to_json.py
python dataset/train/data_pro/gene_json_stage2_pred.py
```

🚀 Train the mask-refinement model with the predicted-mask JSON:

```bash
cd stage2_FLUX_ControlNet
bash train.sh
# model_with_gt_mask is the checkpoint trained in Step 4: Train Stage 2 with GT Masks.
```

## Evaluation

MIG-Eval is available on [🤗 Hugging Face](https://huggingface.co/datasets/xiaoyu1104/MIG-Eval).

Before evaluation, download [🤗 sam2_hiera_large.pt](https://huggingface.co/facebook/sam2-hiera-large), [🤗 fg-clip](https://huggingface.co/qihoo360/fg-clip-large), and [🤗 Qwen2-VL-72B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-AWQ), then place them under `evalution/pretrain_model/`.

🧪 Run the evaluation script for IoU, local CLIP, and ACC:

```bash
bash evalution/test.sh
```

## Citation

If you find this project useful, please cite our work. Citation information will be added after the paper is available.

```bibtex
@article{instancecontrol,
  title   = {InstanceControl: Controllable Complex Image Generation without Instance Labeling},
  author  = {InstanceControl Contributors},
  journal = {arXiv preprint},
  year    = {2026}
}
```
