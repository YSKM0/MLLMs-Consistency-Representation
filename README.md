# Consistency-Driven Representation Learning for Spatial Reasoning in Multimodal Large Language Models

**[Hanwen Liu](https://github.com/YSKM0) · Master’s thesis · ETH Zürich · November 2025 – May 2026**  
Computer Vision and Geometry Group & PRS Group

**Can consistency across views help a multimodal language model reason about 3D scenes?**

This project investigates training objectives that encourage a 7B-parameter LLaVA-Video model to learn spatially consistent representations from multiple views of an indoor scene. It develops three approaches—temporal cycle consistency, geometry-guided correspondence distillation, and occupancy-driven object-level consistency—and evaluates their effect on 3D question answering, captioning, and grounding.

**Main result:** temporal cycle consistency raises SQA3D Exact Match from **46.5 to 52.1 (+5.6 points; approximately 12% relative)**. Occupancy-driven consistency reaches **50.8 (+4.3 points; approximately 9% relative)**.

> This is the thesis research repository, built on [Ross3D](https://github.com/Haochen-Wang409/ross3d) and [Video-3D-LLM](https://github.com/LaVi-Lab/Video-3D-LLM). The consistency objectives, occupancy annotation pipeline, and associated integration are thesis contributions. The underlying LLaVA-Video architecture and Ross3D reconstruction methods come from prior work.

[Goal](#research-goal) · [Results](#results) · [Methods](#proposed-methods) · [Code](#implementation-guide) · [Setup](#getting-started) · [Acknowledgements](#acknowledgements-and-citation)

## Research goal

A model can describe individual images without maintaining a consistent representation of the same scene across viewpoints. Spatial reasoning requires more: relating observations, preserving object identity, and understanding where objects are relative to the observer and to one another.

The thesis asks whether **consistency in the model’s internal visual representations can serve as an inductive bias for spatial reasoning**. Rather than designing a new language-model backbone, it adds auxiliary objectives to decoder-side visual representations during multimodal instruction tuning.

The research explores two levels of representation:

- **Patch level:** learn correspondences across views through cycle consistency or explicit 3D correspondence supervision.
- **Object level:** use patch occupancy to aggregate features into object-instance representations, then supervise their temporal consistency and geometry.

## Results

**SQA3D** is a situated 3D question-answering benchmark that tests spatial reasoning from an observer’s position and orientation in an indoor scene.

| Thesis experiment | SQA3D Exact Match ↑ | Change vs. baseline |
| :--- | ---: | ---: |
| Baseline used in the thesis | 46.5 | — |
| **Temporal cycle consistency** | **52.1** | **+5.6 points** |
| Occupancy-driven object-level consistency | 50.8 | +4.3 points |
| Geometry-guided correspondence distillation | 43.7 | −2.8 points |

These are results from the thesis experiments, rather than a reproduction of the upstream Ross3D paper’s result table. Each method is a separate experimental variant; the gains are not additive.

The three approaches were evaluated across five benchmarks:

| Benchmark | Task |
| :--- | :--- |
| SQA3D | Situated 3D question answering |
| ScanQA | 3D question answering |
| Scan2Cap | Object-centric captioning |
| ScanRefer | Language-guided 3D grounding |
| Multi3DRefer | Language-guided grounding of multiple objects |

### What the experiments show

Temporal cycle consistency produced the strongest SQA3D result among the proposed variants. Object-level consistency also improved over the baseline, supporting occupancy-weighted object representations as a useful alternative to patch-level supervision.

Geometry-guided distillation did not improve SQA3D. Its analysis highlighted the ambiguity of patch-averaged world coordinates: a patch may mix different objects, surfaces, and background, so its average coordinate may be a poor correspondence target. This motivated exploration of supervision at the object-instance level.

The thesis also discusses a separate evaluation limitation: language-sensitive metrics can penalize reasonable answers that differ in wording or granularity, and benchmark scores do not fully isolate spatial understanding. This does not establish that evaluation sensitivity caused the geometry variant’s lower score.

## Proposed methods

All three methods operate within the existing multimodal training pipeline and are trained alongside the language-model objective.

### 1. Temporal cycle consistency

**Goal:** improve spatial reasoning by encouraging visual patch representations to remain consistent across views.

The method reconstructs frame-aligned patch features from the decoder’s packed visual tokens and forms soft correspondence matrices between selected frames using feature similarity. Forward transitions move from an initial frame through later frames; backward transitions return along the sequence.

For a forward-and-backward transition product \(M\), the objective rewards returning to the original patch:

$$
\mathcal{L}_{\mathrm{cycle}}
= -\frac{1}{P}\sum_{i=1}^{P}\log(M_{ii}+\epsilon).
$$

Here, \(P\) is the number of patches in the starting frame. The forward and backward matrices are separately normalized from shared feature similarities; the backward matrix is not an independently learned inverse.

The objective encourages round-trip consistency. It does not, by itself, mathematically guarantee semantic correspondences.

**Implementation:** [compute_cycle_consistency_loss](ross3d/model/ross3d_arch.py).

### 2. Geometry-guided correspondence distillation

**Goal:** test whether explicit 3D geometry can provide a useful teacher signal for learned visual correspondences.

The method builds teacher correspondence distributions from patch-level world coordinates and trains appearance-based student distributions to match them. The implementation uses normalized correspondence logits and teacher-to-student KL losses in both frame directions, with row and column distributions.

This tests a different source of supervision from round-trip consistency: correspondence targets derived from 3D structure.

**Research finding:** more explicit geometric supervision did not automatically translate into better spatial-question-answering performance. The quality and semantic meaning of the supervision unit matter, motivating the object-centric approach below.

**Implementation:** [compute_cycle_consistency_loss_v2](ross3d/model/ross3d_arch.py). Despite its historical function name, the current v2 implementation returns the weighted geometry-distillation loss; it does not add the original cycle loss.

### 3. Occupancy-driven object-level consistency

**Goal:** preserve object identity and spatial structure across views using object instances as the unit of supervision.

#### RGB-D annotation pipeline

The thesis introduces an independently implemented pipeline that combines ScanNet scene geometry, depth, camera parameters, instance segmentations, and object aggregation information to generate:

- **Visible-object annotations:** object IDs, labels, projected boxes, centers, and visibility information.
- **Patch-occupancy annotations:** the fraction of pixels in each visual patch belonging to each object instance.

Occupancy provides a soft assignment: a patch can contribute to several objects in proportion to their visible support. The annotations are aligned with the visual encoder’s resized input and patch grid.

#### Occupancy-weighted object representations

For projected patch feature \(v_{f,i}\) and occupancy weight \(w_{f,i,o}\), an object representation is constructed as:

$$
z_{f,o}
= \operatorname{LayerNorm}\left(
\frac{\sum_i w_{f,i,o}v_{f,i}}
{\sum_i w_{f,i,o}+\epsilon}
\right).
$$

Persistent object IDs align these representations across frames. A presence mask handles objects that are absent from a particular view.

#### Temporal and geometry objectives

- **Temporal consistency:** align observations of the same instance across views, while using category-aware margins to distinguish other instances. Leave-one-out object prototypes avoid matching an observation only to itself.
- **Geometry-aware supervision:** predict object support over patches, projected boxes, center distributions, and center visibility from object-conditioned patch features.

The combined auxiliary objective is:

$$
\mathcal{L}_{\mathrm{occ}}
= \lambda_{\mathrm{temp}}\mathcal{L}_{\mathrm{temp}}
+ \lambda_{\mathrm{geo}}\mathcal{L}_{\mathrm{geo}}.
$$

**Implementation:** [extract_occupancy_object_embeddings, compute_occupancy_temporal_loss, and compute_occupancy_geometry_loss](ross3d/model/ross3d_arch.py).

## Training and engineering

The final reported variants were trained for **one complete epoch on 223,128 examples**, primarily using **four NVIDIA A100 GPUs**, with approximately **35 hours per run**.

The implementation integrates custom PyTorch objectives with the existing model, data loader, and distributed training stack. Engineering work included:

- Integrating DeepSpeed ZeRO-3 to eliminate GPU out-of-memory failures.
- Resolving distributed hangs by keeping occupancy-related parameterized execution consistent across ranks.
- Stabilizing `torch.compile` by isolating dynamic, data-dependent operations from compiled regions.
- Aligning packed decoder tokens, frame-level patch features, and geometry or occupancy annotations.

The repository also contains exploratory farthest-view sampling and camera-pose utilities. These are supporting experiments rather than the principal thesis methods.

## Implementation guide

| Component | Location |
| :--- | :--- |
| Patch extraction, cycle loss, geometry distillation, object embeddings, and occupancy losses | [ross3d/model/ross3d_arch.py](ross3d/model/ross3d_arch.py) |
| Integration of auxiliary losses into the model forward pass | [ross3d/model/language_model/ross3d_qwen.py](ross3d/model/language_model/ross3d_qwen.py) |
| Training arguments and data integration | [ross3d/train/train_3d.py](ross3d/train/train_3d.py) |
| Frame selection and occupancy / visible-object annotation loading | [ross3d/video_utils.py](ross3d/video_utils.py) |
| Training mixture | [scripts/3d/train/video3dllm_223k.yaml](scripts/3d/train/video3dllm_223k.yaml) |
| Training launch scripts | [scripts/3d/train/](scripts/3d/train/) |
| DeepSpeed configuration | [scripts/3d/zero3.json](scripts/3d/zero3.json) |
| Evaluation launch scripts | [scripts/3d/eval/](scripts/3d/eval/) |
| Cluster-specific launch examples | [cscs_script/](cscs_script/) |

### Research artifact status

The model-side consistency methods and annotation-loading integration are present in this repository. The independently written occupancy-generation scripts, `preprocess_bbox.py` and `preprocess_occupancy.py`, are not included in the current code snapshot. Occupancy training therefore requires precomputed annotations or those scripts.

The thesis PDF, final presentation, and thesis-trained model checkpoints are not bundled in this snapshot. The upstream Ross3D checkpoints linked in the [archived README](README.upstream.md) are baseline resources, not checkpoints for the thesis variants.

## Getting started

### Installation

Use a Linux environment with a compatible CUDA/PyTorch setup. The repository retains the inherited package dependencies; check [pyproject.toml](pyproject.toml) and the environment files when preparing your environment.

```bash
git clone https://github.com/YSKM0/Ross3D-Euler.git
cd Ross3D-Euler

conda create -n ross3d-thesis python=3.10 -y
conda activate ross3d-thesis
pip install --upgrade pip
pip install -e .
pip install flash-attn --no-build-isolation
```

### Data and model preparation

1. Follow the [data preparation instructions](scripts/3d/preprocessing/README.md) for the inherited ScanNet and benchmark processing workflow.
2. Prepare the base model and any adapters required by your selected configuration. The [archived upstream README](README.upstream.md) documents the original Ross3D resources.
3. For occupancy experiments, provide the per-frame `patch_occupancy/<frame>_patch_occupancy.json` and `visible_bboxes/<frame>_visible_bboxes.json` annotations expected by [video_utils.py](ross3d/video_utils.py).
4. Update the dataset, checkpoint, vision-tower, and output paths in the selected launch script for your environment.

### Selecting a method

The relevant switches are defined in [train_3d.py](ross3d/train/train_3d.py):

| Variant | Main switches |
| :--- | :--- |
| Temporal cycle consistency | `--cycle_consist True` |
| Geometry-guided correspondence distillation | `--cycle_consist_v2 True` |
| Occupancy temporal objective | `--enable_occ_temp_loss True --occ_temp_loss_weight <weight>` |
| Occupancy geometry objective | `--enable_occ_geom_loss True --occ_geom_loss_weight <weight>` |

These switches identify code paths, rather than complete final-experiment recipes. Set the corresponding weights, geometry options, frame sampling, and inherited reconstruction settings for the experiment you intend to run. Occupancy loss weights default to zero.

The [cycle launch script](scripts/3d/train/train_cycle.sh) and [cluster examples](cscs_script/) are starting points. They contain environment-specific paths and differing GPU configurations; they should be adapted before execution.

### Evaluation

After preparing data, checkpoints, and the evaluation environment:

```bash
bash scripts/3d/eval/eval_all.sh <checkpoint_path> 32
```

This dispatches the five benchmark evaluation scripts. The current scripts explicitly request eight GPUs; adjust their `CUDA_VISIBLE_DEVICES` and `--n_gpu` settings to match your allocation. Their dataset and output paths may also need adjustment.

## Acknowledgements and citation

This thesis extends existing open-source work:

- [Ross3D](https://github.com/Haochen-Wang409/ross3d): inherited codebase, reconstruction components, and baseline resources.
- [Video-3D-LLM](https://github.com/LaVi-Lab/Video-3D-LLM): underlying 3D vision-language framework and data-processing foundation.
- LLaVA-Video / Qwen2 and SigLIP: pretrained multimodal, language, and vision components.
- [ScanNet](https://github.com/ScanNet/ScanNet), [SQA3D](https://github.com/SilongYong/SQA3D), [ScanQA](https://github.com/ATR-DBI/ScanQA), [Scan2Cap](https://github.com/daveredrum/Scan2Cap), [ScanRefer](https://github.com/daveredrum/ScanRefer), and [Multi3DRefer](https://github.com/3dlg-hcvc/M3DRef-CLIP): scene data and evaluation benchmarks.

The original Ross3D README, including its paper citation and resource links, is preserved in [README.upstream.md](README.upstream.md).

When using the inherited Ross3D components, please cite the original work:

```bibtex
@article{wang2025ross3d,
  title={Ross3D: Reconstructive visual instruction tuning with 3D-awareness},
  author={Wang, Haochen and Zhao, Yucheng and Wang, Tiancai and Fan, Haoqiang and Zhang, Xiangyu and Zhang, Zhaoxiang},
  journal={arXiv preprint arXiv:2504.01901},
  year={2025}
}
```

## License

The existing [Apache 2.0 license](LICENSE) is retained. Datasets, pretrained checkpoints, and other third-party resources remain subject to their respective licenses and usage terms.
