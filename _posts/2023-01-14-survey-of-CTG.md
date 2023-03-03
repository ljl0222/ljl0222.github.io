---
title: A Survey of CTG
author: ljl
date: 2023-01-14
categories: [NLP]
tags: [Transformer, Diffusion-Model, Controllable Text Generation]
math: true
mermaid: true
---


参考文章：[A Survey of Controllable Text Generation using Transformer-based Pre-trained Language Models](https://arxiv.org/pdf/2201.05337.pdf)



## 写在前面的前面

新年快乐，这段时间需要加油啦！

![Desktop View](/assets/img/posts/2023-01-14-survey-of-CTG.md/jiayou.png)


可控文本生成是现在产生的一个比较新颖的方向，其主要的建模目标在于，将我们给出的条件或者称之为属性放入我们的输入当中，从而建模$P(Y\mid{X,c})$这样一个目标，最终使得我们的输出可以在给定的输入和属性下产生我们想要的，具有属性c的文本内容。

笔者注意到，大部分网络上整理的内容都是完全基于A Survey of Controllable Text Generation using Transformer-based Pre-trained Language Models(后面简称Survey)这篇综述文章的。

诚然，这篇文章的确是CTG的有关内容在网络上为数不多的综述性质的文章，针对预训练模型的有关内容也讲解的较为全面。然而实际上，在近年，尤其是2022年间，扩散模型在CTG上的应用其实是十分广泛的，而基于VAE的方法也层出不穷，由于这些内容并不是基于预训练的模型，综述内容并没有覆盖到这些，因此本文旨在整理有关的CTG的内容以及文章，尝试在整理这篇博客的过程当中寻找到各个文章之间的内容联系，划分出大致的文章类别，从合适的角度去俯瞰这些工作。

说了一把子废话，画了一些大饼，现在开始尝试整理。我仍然将Survey文章作为重要的参考文献和出发点。

## 基于预训练的CTG模型

正如很多文章所说的一样，预训练模型正在大放异彩，因此尝试在CTG的任务下将目光聚焦在预训练模型是很正常的。很多有关的技术博客都说明了，将已经在大规模语料预训练过的预训练模型应用在CTG任务当中是存在难度的，从大的方向上来说，目前我们主要有三种使用方法：

1. 改良派：通过微调的方法，类似原先的方式，使用一些技巧在预训练的基础上继续Fine-tuning。
2. 革命派：就是有钱！打破原先的预训练模型，另起炉灶，重新训练一个大规模的预训练模型。
3. 保守派：摆烂，我就想用预训练模型，因此只在预训练的解码过程当中使用一些额外的方法和模型来控制生成的内容。

### 改良派

改良派实际上是沿用了预训练时代当中一个比较通用的范式，也就是Fine-Tuning范式，我们使用一部分特定的语料，将可控文本生成的任务当中的可控部分，寄托于我们所给出的特定的语料。

对于改良派，Survey当中也给出了大概三种分类：

1. 基于adapter的方法：所谓adapter，其实就是在模型当中的某层添加少量的参数，我们冻结PLM的参数，将我们可控的希望寄托在adapter上。直观的感觉上来讲，对于多种其他的任务来说，我们也可以应用这种方法进行类似的处理，将其他任务的特性寄托在adapter上。
2. 基于prompt的方法：基于prompt的方法实际上还是比较丰富的，简单的来说，我们在输入当中使用一定的token来对PLM当中添加额外的控制。与adapter相比，prompt的方法实际上是添加在输入上，而adapter实际上是添加在模型当中的。（顺便一提，与prefix-tuning相比，prompt的方法仍然是给定了prompt去微调PLM，而prefix-tuning则是完全微调trainable的添加在输入的前缀）

![Desktop View](/assets/img/posts/2023-01-14-survey-of-CTG.md/prompt.png)

3. 基于强化学习的方法：通过奖励机制来反馈控制条件的实现，有人使用强化学习的方法，引导GPT2朝着指定的条件方向（即目标类）生成文本内容，具体来说，就是在GPT2的softmax和argmax函数之间增加了一个额外的RL阶段，根据RL的奖励信号朝着目标标签更新，即通常来讲，这里是结合RL使用微调的方法。

#### Fine-Tuning

在这个类别下的是纯纯什么都没干的，好像看起来看起来就是在人家的框架上微调了一下。

Survey当中列举的任务主要是AMR-to-text（抽象语义到文本任务）。

<table>
<tr>
    <th>文章名称</th>
    <th>模型</th>
    <th>目标任务</th>
    <th>备注</th>
</tr>
<tr>
    <td>Text-to-Text Pre-Training for Data-to-Text Tasks</td>
    <td>T5</td>
    <td>Data-to-Text</td>
    <td>仅仅实现了一个非结构文本的<br>转换，直接在T5的基础上微调</td>
</tr>
<tr>
    <td>DART: Open-Domain Structured Data Record <br>to Text Generation</td>
    <td>BART,T5</td>
    <td>Data-to-Text</td>
    <td>构造了一个新数据集DART，<br>合并了E2E和WebNLG，<br>还取了很多别的数据</td>
</tr>
<tr>
    <td>Investigating Pretrained Language Models <br>for Graph-to-Text Generation</td>
    <td>BART,T5</td>
    <td>Graph-to-Text</td>
    <td>调查并比较了BART和T5<br>这两个PLMs，用于图到文本生成</td>
</tr>
</table>

#### Adapter

在这个类别下的，通常是在原先的预训练模型的基础上添加了一些结构，并且在这些结构的基础上冻结原先大部分PLMs的参数，使用adapter来进行属性的控制。

<table>
<tr>
    <th>文章名称</th>
    <th>模型</th>
    <th>目标任务</th>
    <th>备注</th>
</tr>
<tr>
    <td>Technical Report: Auxiliary Tuning and its <br>Application to Conditional Text Generation</td>
    <td>自回归的<br>Transformer</td>
    <td>Data-to-Text</td>
    <td>提出了一种新的微调范式<br>Auxiliary Tuning，流畅性<br>更好</td>
</tr>
<tr>
    <td>Structural Adapters in Pretrained Language <br>Models for AMR-to-text Generation</td>
    <td>自回归的<br>Transformer</td>
    <td>Graph-to-Text</td>
    <td>提出了STRUCTADAPT，建<br>模图结构</td>
</tr>
<tr>
    <td>DIALOGPT : Large-Scale Generative<br>Pre-training for Conversational Response <br>Generation</td>
    <td>GPT2</td>
    <td>Dialogue Generation</td>
    <td>构造了MMI评分函数，惩<br>罚乏味假设</td>
</tr>
</table>

#### Reinforcement Learning

强化学习在可控文本生成任务当中，其主要的方法是通过反馈控制条件是否或如何实现，从而作为对预训练模型微调的奖励，同时，一般为了保证生成文本的流畅性，还会在奖励函数当中添加一个惩罚项。

<table>
<tr>
    <th>文章名称</th>
    <th>模型</th>
    <th>目标任务</th>
    <th>备注</th>
</tr>
<tr>
    <td>Data Boost: Text Data Augmentation<br>Through Reinforcement Learning<br>Guided Conditional Generation</td>
    <td>GPT2</td>
    <td>分类任务（生成部<br>分作为数据增强）</td>
    <td>增加了额外的RL阶段，更新<br>PLM的隐层参数</td>
</tr>
<tr>
    <td>Learning to summarize from human feedback</td>
    <td>GPT3</td>
    <td>英文摘要</td>
    <td>结合了人工评估，使用强<br>化学习的方法微调</td>
</tr>
<tr>
    <td>Controllable Neural Story Generation <br>via Reinforcement Learning</td>
    <td>LSTM</td>
    <td>可控故事生成</td>
    <td>在所有时间步产生中间奖励，<br>反向传播到语言模型当中</td>
</tr>
</table>

### 革命派

正如我们前面所介绍的，改革派直接进行微调，或者借助adapter等结构来在已有的预训练模型的基础上进行微调，而革命派则直接改变原有的预训练模型的架构，或者干脆从头训练一个预训练模型，从而使其更加适配下游任务。这样的方法有望大幅提升文本生成的质量和可控性，但是存在标记数据不足和计算资源消耗大的局限性。

<table>
<tr>
    <th>文章名称</th>
    <th>模型</th>
    <th>目标任务</th>
    <th>备注</th>
</tr>
<tr>
    <td>CTRL: A CONDITIONAL TRANSFORMER<br>LANGUAGE MODEL FOR CONTROLLABLE<br>GENERATION</td>
    <td>Transformer</td>
    <td>各种可控<br>生成任务</td>
    <td>在文本语料库前添加了控<br>制代码，重新训练</td>
</tr>
<tr>
    <td>POINTER: Constrained Progressive<br>Text Generation via Insertion-based<br> Generative Pre-training</td>
    <td>Transformer</td>
    <td>可控文本<br>生成</td>
    <td>修改了transformer结构，<br>先产生约束词，再插入其他词</td>
</tr>
<tr>
    <td>Parallel Refinements for Lexically<br>Constrained Text Generation with<br>BART</td>
    <td>BART</td>
    <td>可控文本<br>生成</td>
    <td>修改了BART结构，引导模型<br>预测替换和插入的token位置</td>
</tr>
<tr>
    <td>CoCon: A Self-Supervised Approach<br>for Controlled Text Generation</td>
    <td>GPT</td>
    <td>可控文本<br>生成</td>
    <td>将控制块放入GPT模型当中</td>
</tr>
</table>

### 保守派

直接采用后处理的方法，即我们仅仅在解码阶段应用一定的策略，使我们产生的语句可以获得想要的属性和内容。显而易见的是，这样的方法我们可以使用较小的计算资源来获得不错的结果，同时也最大限度的激活了预训练模型的潜力。

<table>
<tr>
    <th>文章名称</th>
    <th>模型</th>
    <th>目标任务</th>
    <th>备注</th>
</tr>
<tr>
    <td>Plug and Play Language Models:<br>A Simple Approach to Controlled<br>Text Generation</td>
    <td>GPT2</td>
    <td>各种可控<br>生成任务</td>
    <td>使用一个属性判别器来<br>指导文本生成</td>
</tr>
<tr>
    <td>FUDGE: Controlled Text Generation<br>With Future Discriminators</td>
    <td>GPT2</td>
    <td>各种可控<br>生成任务</td>
    <td>将未来产生的token添<br>加到输出当中判断属性</td>
</tr>
</table>


## 非预训练的CTG模型

### VAE & GAN

本部分内容，尤其是有关VAE以及GAN的内容主要参考于An Overview on Controllable Text Generation via Variational Auto-Encoders这篇文章，后文称作Overview。

Overview当中主要从训练的方法来对不同的应用于可控文本生成的自编码器模型进行了有关的分类，主要分为有监督的，半监督的以及无监督的方法。

#### 监督的方法

对于将VAE应用于可控文本生成任务，最直观的做法就是我们将给出的所有标签都合并到生成过程当中。也因此我们引出了CVAE等模型结构。

<table>
<tr>
    <th>文章名称</th>
    <th>目标任务</th>
    <th>备注</th>
</tr>
<tr>
    <td>Improve Diverse Text Generation by<br>Self Labeling Conditional Variational <br>Auto Encoder</td>
    <td>各种可控<br>生成任务</td>
    <td>将x注入z，回避了潜在<br>的KL崩溃问题</td>
</tr>
<tr>
    <td>FUDGE: Controlled Text Generation<br>With Future Discriminators</td>
    <td>各种可控<br>生成任务</td>
    <td>将未来产生的token添<br>加到输出当中判断属性</td>
</tr>
</table>

#### 半监督的方法

#### 无监督的方法

#### 评价指标的分类

### Diffusion Model

## 总结

## 常见的生成任务和数据集

<table>
<tr>
    <th>生成任务</th>
    <th>数据集</th>
    <th>备注</th>
</tr>
<tr>
    <td rowspan="5">Data-to-Text</td>
    <td>ToTTo</td>
    <td>自然语言描述的维基百科单元格</td>
</tr>
<tr>
    <td>MultiWoz</td>
    <td>包含10K个人际对话的语料库，用于开发面向任务的对话系统</td>
</tr>
<tr>
    <td>Cleaned E2E</td>
    <td>对话行为含义表示(MR)和餐厅领域的自然语言引用到文本描述</td>
</tr>
<tr>
    <td>DART</td>
    <td><三元组，句子>到文本描述</td>
</tr>
<tr>
    <td>Reddit</td>
    <td>被抓取的Reddit自2005年至2017年的评论链</td>
</tr>
<tr>
    <td rowspan="3">Graph-to-Text</td>
    <td>WebNLG</td>
    <td>主题-对象-谓词三元组的图和文本描述</td>
</tr>
<tr>
    <td>AMR</td>
    <td>谁对谁做什么”的有根有向图</td>
</tr>
<tr>
    <td>AGENDA</td>
    <td>论文标题，一个KG，以及相应的摘要</td>
</tr>
<tr>
    <td>英文摘要</td>
    <td>TL;DR</td>
    <td>300万篇不同主题的帖子</td>
</tr>
<tr>
    <td>故事生成</td>
    <td>CMU movie<br>summary corpus</td>
    <td>电影故事语料库，种类很多</td>
</tr>
<tr>
    <td rowspan="3">1</td>
    <td>2</td>
    <td>3</td>
</tr>
<tr>
    <td>2</td>
    <td>3</td>
</tr>
<tr>
    <td>2</td>
    <td>3</td>
</tr>
<tr>
    <td>1</td>
    <td>2</td>
    <td>3</td>
</tr>

</table>

## 参考文献

