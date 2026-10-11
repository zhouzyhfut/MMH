# Differential Mental Disorder Detection with Psychology-Inspired Multimodal Stimuli

##  ACM MM 2026

---

## 🌟 Overview

Differential diagnosis of mental disorders is difficult because several conditions
share overlapping symptoms, while most existing public datasets are built for a
**single** disorder and elicit behaviour with a narrow paradigm (interview or
reading only).

This work designs a **psychology-inspired multimodal elicitation paradigm** — five
tasks covering emotional reactivity, cognitive load, social evaluation and
cross-modal integration — and collects **MMH**, a large-scale multimodal mental
health dataset covering **depression (MD)**, **anxiety (ANX)** and
**schizophrenia (SC)** together with healthy controls (**HC**), with every
diagnostic label clinically verified by licensed psychiatrists (928 participants).

On top of MMH we propose **PMLF**, a paradigm-aware multimodal learning framework:

| Stage | Directory | What it does |
| --- | --- | --- |
| 1. Prompt-guided pretraining | `pretrain_clip/` | A multimodal LLM (Gemini-2.5-pro) generates a sample-level description of the context induced by each elicitation task; the video encoder is aligned with these descriptions by cross-modality contrastive learning against a frozen CLIP text encoder. |
| 2. Multimodal joint training | `CLIPtrans/` | Audio (13-d MFCC), text (768-d BERT-base-Chinese) and the pre-trained video encoder (714-d OpenFace features) are fused with cross-attention and optimised with cross-entropy plus audio↔text / audio↔video contrastive losses. |

---

## 📦 Data Availability

The **MMH** dataset is available on Hugging Face:
**[HFUT-MMH-Lab/multimodal_mental_health_dataset_MMH](https://huggingface.co/datasets/HFUT-MMH-Lab/multimodal_mental_health_dataset_MMH)**


## 🛠️ Environment

```bash
conda create -n mmh python=3.12 -y
conda activate mmh

pip install -r requirements.txt
```

---

## 🚀 Training and Evaluation

```bash
# 1) tell the code where the data is
export MMH_PRETRAIN_DATA_ROOT=/path/to/pretraindata   # stage-1 data root
export MMH_DATA_ROOT=/path/to/data                    # stage-2 data root

# 2) stage 1 — pretrain the video encoder        (pretrain_clip/pretrain.py)
cd pretrain_clip
python pretrain.py --lr 0.001 --gpu 0
#    -> pretrain_clip/save_model/<epoch>.t

# 3) stage 2 — train the multimodal classifier   (CLIPtrans/train.py)
cd ../CLIPtrans
python train.py --lr 0.001 --gpu 0 \
    --pretrain_path ../pretrain_clip/save_model/28.t
#    -> save_model/Triplet/base_n8_lr_0.001_seed_0_best.t
#       save_model/Triplet/base_n8_lr_0.001_seed_0_bestf1.t

# 4) evaluate a trained checkpoint               (CLIPtrans/test.py)
python test.py --lr 0.001 --gpu 0 \
    --pretrain_path ../pretrain_clip/save_model/28.t \
    --load_path save_model/Triplet/base_n8_lr_0.001_seed_0_bestf1.t
```


---

## 📂 Data Layout

Paths inside every list file are **relative to the data root**; write one list file
per modality and split.

```
<MMH_PRETRAIN_DATA_ROOT>/                     # stage 1
├── pretrainpath/trainvideo.txt               # /0/07011970/video_feature/video_01/video_01.csv
├── pretrainpath/traintext.txt                # /0/07011970/video_feature/video_01/video_01.txt
└── 0/07011970/video_feature/video_01/
    ├── video_01.csv                          # OpenFace features, one row per frame
    └── video_01.txt                          # MLLM description (the CLIP text)

<MMH_DATA_ROOT>/                              # stage 2
├── trainpath/                                # {split}{audio,text,video}path.txt
├── valpath/                                  #   lines:  <relative path> <label>
├── testpath/
└── 0/07011970/
    ├── combined_mfcc.npy                     # audio      (13-d MFCC)
    ├── 07011970.npy                          # text       (768-d BERT-base-Chinese)
    └── video.csv                             # video      (714-d OpenFace)
```


---

## 📚 Citation

If you find this work useful or inspiring for your research, please consider
citing our paper:

> Zhou, Z., Wu, J., Lei, Z., Guo, J., Yu, Z., Chu, Y., Zhang, X., Zhao, Q.,
> Wang, Q., Hao, S., Guo, Y., & Hong, R. (2026).
> **Differential Mental Disorder Detection with Psychology-Inspired Multimodal
> Stimuli.**
> *Proceedings of the ACM International Conference on Multimedia (ACM MM)*.
> https://arxiv.org/abs/2604.02798

BibTeX:

```bibtex
@misc{zhou2026differentialmentaldisorderdetection,
      title={Differential Mental Disorder Detection with Psychology-Inspired Multimodal Stimuli},
      author={Zhiyuan Zhou and Jingjing Wu and Zhibo Lei and Junyu Guo and Zhongcheng Yu and Yuqi Chu and Xiaowei Zhang and Qiqi Zhao and Qi Wang and Shijie Hao and Yanrong Guo and Richang Hong},
      year={2026},
      eprint={2604.02798},
      archivePrefix={arXiv},
      primaryClass={cs.MM},
      url={https://arxiv.org/abs/2604.02798},
}
```

---

## 📄 License

Released under the **MIT License** — see [`LICENSE`](LICENSE).

---
