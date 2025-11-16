# DMaS-LLaMa-Lite: 1.7B-Parameter LLaMa Model Training Code

This repository contains the training scripts and configuration files for **DMaS-LLaMa-Lite**, a 1.7B-parameter model inspired by the LLaMa architecture and trained from scratch on approximately 20 billion tokens of curated high-quality data.

The overall training code is modified from [BuildNanoGPT](https://github.com/karpathy/build-nanogpt), while the LLaMa implementation is adapted from [hengjiUSTC's learn-llm](https://github.com/hengjiUSTC/learn-llm).

---

## Model Overview

DMaS-LLaMa-Lite is a **1.7B-parameter LLaMa-based language model** pretrained with the following key highlights:
- **High-Quality Training Data**: Curated from the FineWeb-Edu dataset, emphasizing educational and coherent content.
- **Training Stability**: Insights include the importance of optimizer state restoration and managing hardware transitions.
- **Efficient Performance**: Competitive downstream task results achieved with significantly fewer tokens than comparable models.

For more details, refer to our paper:  
> **Experience of Training a 1.7B-Parameter LLaMa Model From Scratch**  
> *Miles Q Li, Benjamin Fung, Shih-Chia Huang* (2024)  
> [arXiv preprint link](https://arxiv.org/abs/2412.13335)  

---

## Citation

If you use this repository or the pre-trained model, please cite our work:

```bibtex
@INPROCEEDINGS{11228044,
  author={Li, Miles Q. and Fung, Benjamin C. M. and Huang, Shih-Chia},
  booktitle={2025 International Joint Conference on Neural Networks (IJCNN)},
  title={Training Dynamics of a 1.7B LLaMa Model: A Data-Efficient Approach},
  year={2025},
  volume={},
  number={},
  pages={1-10},
  keywords={Training;Analytical models;Refining;Benchmark testing;Throughput;Data models;Hardware;Stability analysis;Trajectory;Tuning},
  doi={10.1109/IJCNN64981.2025.11228044}}
```

---

## Resources
- **Training Code**: [GitHub Repository](https://github.com/McGill-DMaS/DMaS-LLaMa-Lite-Training-Code)
- **Pre-trained Checkpoints**: [HuggingFace Collection](https://huggingface.co/collections/McGill-DMaS/dmas-llama-lite-6761d97ba903f82341954ceb)

---

## Acknowledgments

This work builds on open-source projects, including:
- [BuildNanoGPT](https://github.com/karpathy/build-nanogpt)
- [learn-llm](https://github.com/hengjiUSTC/learn-llm)

We thank the community for their valuable contributions and tools that make this research possible.

