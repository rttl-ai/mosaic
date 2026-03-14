# MOSAIC

MOSAIC is a multi-stage framework for domain adaptation of sentence embedding
models via domain-specific vocabulary expansion and joint MLM–contrastive training.

This repository accompanies the paper:

**MOSAIC: Masked Objective with Selective Adaptation for In-domain Contrastive Learning**  
(EACL 2026, Findings)

📄 arXiv: https://arxiv.org/abs/2510.16797

![MOSAIC overview](figures/approach.png)

## Repository Status

Code cleanup and documentation are currently in progress.
The full training and evaluation code, along with pretrained models,
will be released shortly.

## Code Structure

This repository is based on the open-source
[`contrastors`](https://github.com/nomic-ai/contrastors) framework and contains
a small number of targeted modifications required to implement MOSAIC.

A detailed summary of changes relative to the original codebase is provided
in `MOSAIC_CHANGES.md`.

## Usage (forthcoming)

1. Install dependencies following the original `contrastors` instructions.
2. Use the configuration files provided in `configs/`.
3. Run the training and evaluation scripts in `scripts/`.

Detailed usage instructions will be added as the code is finalized.

## License

This project is released under the Apache 2.0 License, consistent with the
original `contrastors` codebase.
