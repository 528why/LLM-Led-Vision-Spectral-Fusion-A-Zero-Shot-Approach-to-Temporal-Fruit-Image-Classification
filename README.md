# LLM-Led Vision-Spectral Fusion: A Zero-Shot Approach to Temporal Fruit Image Classification


This repository contains the official implementation for the paper: **LLM-Led Vision-Spectral Fusion: A Zero-Shot Approach to Temporal Fruit Image Classification**.

**Authors:** Huyu Wu, Bowen Jia, Xue-Ming Yuan

---

## Abstract

A zero-shot multimodal framework for temporal image classification is proposed, targeting automated fruit quality assessment. The approach leverages large language models for expert-level semantic description generation, which guides zero-shot object detection and segmentation through GLIP and SAM models. Visual features and spectral data are fused to capture both external appearance and internal biochemical properties of fruits. Experiments on the newly constructed **Avocado Freshness Temporal-Spectral (AFTS)** dataset—comprising daily synchronized images and spectral measurements across the full spoilage lifecycle—demonstrate reductions in mean squared error by up to 33% and mean absolute error by up to 17% compared to established baselines. These results validate the effectiveness and generalizability of the framework for temporal image analysis in smart agriculture and food quality monitoring.

---

## 💡 Highlights

- Proposed a multimodal framework integrating **LLMs and vision models** for temporal image classification.
- Developed the **AFTS dataset** with synchronized visual and spectral data for avocado aging analysis.

---


## 🤔 What are Temporally Relevant Images?

Temporally relevant images are visual data that capture the dynamic changes of an object over specific time points. Unlike static images, their classification depends on recognizing subtle, gradual transformations in features like color, texture, and appearance. The temporal context is crucial for accurate interpretation, as these images reflect an evolutionary process.

As illustrated below, these images are distinct from generic, fine-grained, or encoded time-series images. The primary challenge lies in interpreting these gradual visual shifts in complex, real-world scenes. Our AFTS dataset, which tracks avocados as they ripen and spoil, is a prime example of this image type.

![Distinguishing Temporally Relevant Images](./misc/figure1_01.png)
*Comparison of Temporally Relevant Images with generic, fine-grained, and encoded time-series images.*

---


## 🔧 Framework Overview

Our framework integrates an LLM, a Vision-Language Model (VLM), and a spectral encoder to achieve nuanced, context-aware classification. The LLM generates rich semantic prompts that guide GLIP for object detection and SAM for zero-shot segmentation. These visual features are then fused with spectral data for a comprehensive analysis.

![Framework Overview](./misc/figure2_01.png)
*Overview of the proposed framework. An LLM (e.g., GPT-4o, Claude-3.5) generates a knowledge-rich prompt to guide GLIP and SAM for precise object segmentation. Visual and spectral features are then fused for final classification. *

---

## 💾 Dataset

This work introduces the **Avocado Freshness Temporal-Spectral (AFTS)** dataset. It was constructed by documenting 150 avocados daily from a fresh state to complete spoilage.

- **Images:** 12,000 high-resolution (1920×1080) images captured from three angles.
- **Spectral Data:** 5,662 valid spectral curves, with each curve comprising 330 sampling points.

The dataset will be made available soon. Please stay tuned for the download link.

---

## Usage

Code coming soon - installation instructions will be provided.

---
