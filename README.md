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
@inproceedings{LFH25ijcnn,
        author = "M. Q. Li and B. C. M. Fung and S.-C. Huang",
        title = "Training dynamics of a 1.7B {LLaMa} model: a data-efficient approach",
        booktitle = "Proc. of the IEEE International Joint Conference on Neural Networks (IJCNN),
        pages = "",
        address = "Rome, Italy",
        month = "July",
        year = "2025",
        publisher = "IEEE Press",
}
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

