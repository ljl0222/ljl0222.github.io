---
title: The Power of Scale for Parameter-Efficient Prompt Tuning
author: ljl
date: 2022-10-11
categories: [论文阅读, NLP]
tags: [Prompt-Tuning, Prefix-Tuning, Fine-Tuning]
math: true
mermaid: true
---

原文链接：[The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)

他人博客：
- [10min·脑图·第1期 \| 清华博后带你轻松吃透Prompt Tuning顶会论文【OpenBMB论文速读】](https://www.bilibili.com/video/BV18P411E7VK)

- [The Power of Scale for Parameter-Efficient Prompt Tuning及prefix tuning与prompt tuning的区别](https://www.ngui.cc/article/show-349216.html)

## 摘要

- 把Prompt作为一个可训练的向量，固定整个大型的预训练模型PLM，微调Prompt来适配下游任务

- 随着PLM参数规模的增大，Prompt Tuning和Fine Tuning的性能越来越接近

- Prompt Tuning（或者说是基于Soft Prompt的提示微调）可以看作Prefix Tuning的简化版本

## 研究方法与结论

### Fine-Tuning & Prompt Tuning

![Desktop View](/assets/img/posts/2022-10-11-Prompt-tuning/prompt-fine-tuning.png)

- 传统的微调方法需要对所有参数进行调整，针对不同的任务使用不同的微调方法，需要对于不同的任务存储不同的模型副本

- 提示微调，会使得对于不同的任务学习不同的提示，插入到任务的输入当中，肉眼可见的，这样的存储效率会更高，我们不需要保存较大的模型副本

![Desktop View](/assets/img/posts/2022-10-11-Prompt-tuning/%E4%BC%A0%E7%BB%9F%E5%88%B0%E7%8E%B0%E5%9C%A8.png)

- 最左边的是传统方法，给定输入$X$和参数$\theta$，可以得到输出$Y$

- 中间的是GPT3的方法，通过给定输入$X$和一些额外的句子$P$，加上参数$\theta$，可以得到输出$Y$

- 最右边是基于提示微调的方法，在前文的基础上给出一些提示向量$\Delta$，得到输出$Y$

### Prompt-Tuning的影响因素

- 模型本身的参数量（原文当中使用了从T5到T5-XXLarge的模型做实验）

- 使用语言模型的生成，尽力在输出目标当中消除哨兵标记（这点不是很理解，贴上一些别的博客，是和T5模型的预训练任务有关）

![Desktop View](/assets/img/posts/2022-10-11-Prompt-tuning/%E5%88%AB%E4%BA%BA%E7%9A%84%E7%90%86%E8%A7%A3.png)

- 初始化方法

    - 随机初始化
    - 使用预设文本的词向量初始化，类似于设计hard prompt
    - 使用类别词向量初始化，类似于提供选项

## 实验 & 结论

![Desktop View](/assets/img/posts/2022-10-11-Prompt-tuning/labpic.png)

- (a) Prompt的规模越大效果越好

- (b) 基于语义信息的初始化比随机初始化更好

- (c) LM Adaptation对性能提升很显著

- (d) 模型越大，方法效果越接近
