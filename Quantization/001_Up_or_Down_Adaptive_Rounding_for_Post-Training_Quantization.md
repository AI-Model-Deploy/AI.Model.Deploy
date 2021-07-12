# Up or Down? Adaptive Rounding for Post-Training Quantization

## Overview
* Title: Up or Down? Adaptive Rounding for Post-Training Quantization
* Authors: Markus Nagel, Rana Ali Amjad, Mart van Baalen, Christos Louizos, Tijmen Blankevoort
* Link https://arxiv.org/abs/2004.10568 
* Code Repo: https://github.com/quic/aimet 
* Key words: Quantization, Rounding, Post-training quantization, Fine-tuning on local loss.

## 介绍和解读

大多数的CNN和NLP的模型都可以从FLOAT32量化到8-bit的同时，保持相比fp32有一些微小的精度下降，但是运行fps和效率都能得到较大的提升。
所以，大多数工业界的AI inference 芯片基本都支持INT8的量化，如 Intel的VNNI, nvidia的TensorRT, 寒武纪的思远系列。如果我们把量化的比特数从
8bit再降低到4-bit甚至到更低的2-bit，这里边遇到的挑战会大很多了。

这是Qualcomm AI Research发表的一篇文章从Round的角度来分析量化误差，并提出了Adaptive Round的方法来对抗量化误差。从实验结果来看，当weights 量化
到4-bit时，AdaRound和当前的一些方法相比有明显的优势。并且，论文中提到的方法也集成到Qualcomm开源项目[AIMET](https://github.com/quic/aimet)，
感兴趣的同学可以动手试试。

### 文章的核心点
- 对Round带来的误差做了理论和定量的分析，分析Round对于最终task loss的影响。并且抽象成一个优化问题：
   Quadratic Unconstrained Binary Optimization
- AdaRound 的提出，通过per-layer的 fine-tuning来获取对于误差的补偿，这个补偿和每个weights相关的，有的weights会被Round Up，有的weights会被
  Round down。 而且这个补偿是一个可以学习的东西。


### Round 的几种类型
[Rounding](https://en.wikipedia.org/wiki/Rounding) 有挺多种类和不同的行为。Deeplearning量化常用的Rounding，通常有以下几个
- Round down， np.floor
- Round up， np.cell
- RNE (Round Nearest to Even), 当出现0.5 tie的情况，向偶数那边round 
- Stochastic Rounding， 根据概率随机偏向一方，比如说 0.7，有70%的概率round到0， 30%的概率round到1. 这种rounding通常在低精度下的模型训练上
有的较多。

## 实验结果
* Per-layer vs Per-channel 
* 4-bit weights and 8-bit activation ， 为什么不是4w4a？


## 总结



