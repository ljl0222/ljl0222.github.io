---
title: A Survey on In-context Learning
author: ljl
date: 2023-07-30
categories: [NLP]
tags: [Large Language Models, Cross-Cultural Differences for Language Models]
math: true
mermaid: true
---

# In-context Learning(上下文表征学习)

## ICL本身

ICL在LLMs时代非常有用，可以用来评判大语言模型。

简单来说，ICL允许在大模型当中应用一些简单的例子来进行上下文学习，即可以从上下文当中的几个示例来学习，其中，在数学推理(主要应用了CoT)等问题上的能力已经得到了验证。

具体来说，ICL的作用就是给出一些示例，直接丢给大模型，让他去根据其中的潜在模式来给出回答和预测（个人理解是类似few-shot的问答），如下图所示。

![Desktop View](/assets/img/posts/2023-07-30-A-Survey-on-In-context-Learning/ICL-f1.png)

给出ICL的具体定义：ICL是一种范式，这种范式允许语言模型仅仅在被以演示的方法给出少量样本后就可以学习任务。(Incontext learning is a paradigm that allows language models to learn tasks given only a few examplesin the form of demonstration).

ICL的优点：

- 使用可解释的方法来和LLM交互，可以通过改变prompt的构造方法和知识本身来简单的把相关知识融入
- 可解释性强，类似人类学习类比的决策过程
- 无需训练，减少了计算成本

## ICL和大模型相关

比较显著的特性包括，选择的上下文示例的内容，prompt模板乃至上下文示例的顺序都对ICL的性能有影响，此外，尽管普通的GPT-3也可以表现出ICL能力，但是在预训练期间的适应可以显著提升表现能力。

![Desktop View](/assets/img/posts/2023-07-30-A-Survey-on-In-context-Learning/ICL-f2.png)

## ICL和prompt learning & few-shot learning

综述提到了ICL与这两者的关系。严格来说，ICL算是prompt learning的一个子类；而相较于few-shot learning来说，ICL不用参数更新，而是直接使用预训练的LLMs。

# ICL在LLMs当中的各个过程

简单按照前文给出的图来看一下LLMs的各个过程当中ICL发挥的作用，主要分为Training和Inference两个阶段。

## Training

### Model Warmup

 模型预热（model warmup）本身的含义是在预训练的LLMs后，多做一些简单的继续训练（continual training）。

 这里提两点：

 - warmup阶段会修改或者添加LLM的参数
 - 与finetune不同，warmup本身是为了提升ICL能力，而不特异于任务（但是原文后文当中还是混淆了warmup和finetune两个词^^）


#### 有监督的In-context训练

1. MetaICL: Learning to Learn In Context

在大量任务上调整预训练的语言模型，从而提升ICL能力。具体的做法是针对每种不同的任务选取了k+1个实例，从前k个实例当中学习新任务，在最后一个实例上回答问题，计算交叉熵损失。

比较逆天的地方在于，本文在不同的任务集上进行了实验，包括142个不同种类的数据集以及7个不同的实验设置，总共有52个独特的任务。

在我看来，他这里的元训练(meta-trained)的过程和ICL的流程是类似的，本质上也是为了后续LM能够更好地被使用在ICL当中，所以这里进行meta-trained的过程当中也是类似ICL将各种输入数据拼接在一起的构造方法。

对于构造的QA问题对(k个实例(x,y)和最后一个$x_{k+1}$)，还给出了候选项$c_i\in{C}$，对分类问题来说是标签，对问答问题来说是候选项，每组问题返回最大条件概率的标签。

同时，文中还给出了Channel MeatICL的训练方法，本质上是给出k个实例(y,x)和最后一个输出$y_{t+1}$，来反向预测$x_{t+1}$

具体的实验设置和数据集跨度很大，这里仅仅放上最终结果图。

![Desktop View](/assets/img/posts/2023-07-30-A-Survey-on-In-context-Learning/metaICL-res.png)

2. Symbol tuning improves in-context learning in language models

提出了符号调整，我们知道ICL当中的输入输出往往需要通过一些特殊符号分隔，包括特有的标签往往也具有特定含义（例如分类任务当中的negative和positive标签）。因此这里尝试将特殊的标签替换为没有实际意义的标签，例如前面提到的negative和positive替换为foo和bar，这样可以保证模型在不能利用到自然语言找出任务的时候，就必须去学习输入到标签的映射关系。

![Desktop View](/assets/img/posts/2023-07-30-A-Survey-on-In-context-Learning/symol-tuning.png)

3. Finetuned Language Models are Zero-Shot Learners

提出了指令微调的方法，但是感觉非常玄幻，貌似仅仅通过调整任务的描述这样的指令，来提升模型在特定任务上的表现效果。

核心的思路在于，预训练的时候在多种不同的任务类型上进行指令微调，而在推理过程中即使是没见过的任务类型也能取得很好的效果。这样的训练方法可以提升模型遵守给出的指令的能力，因此即使是对于看不到的指令也能够很好的解决问题。

![Desktop View](/assets/img/posts/2023-07-30-A-Survey-on-In-context-Learning/FLAN.png)

值得一提的是，我个人认为这里取得效果的强弱与任务的种类和构造模板的数量密不可分，原文针对多种不同的任务构造了很多不同的模板，而且这里又一次用到了反向问答的技巧。例如针对情感分类任务，要求其直接根据情感产生电影评论。

#### 自监督的In-context训练

1. Improving In-Context Few-Shot Learning via Self-Supervised Training

另外写了一篇简单的博客，主要是采用了四种不同的预训练任务，自动的生成训练数据并且进行自监督的训练。针对不同的训练任务，都采用了不同的数据生成方法。四种方法包括下一句预测，短语预测，掩码词语预测以及分类任务。

2. Pre-Training to Learn in Context

和上一篇文章类似，同样是着眼于预训练任务当中缺少了关于ICL相关内容的训练，因此提出了PICL，相较于上篇工作，本工作在更大规模的数据集上进行了NLP任务的实践，证明了其泛化性更强。

本文同样是针对文本内容进行训练数据的自我构造，提出了一个有趣的观察，许多文本文档当中都包含了很多“内在任务”，LMs对每个段落进行语言建模的过程中，也都隐含的同时执行了相应的内在任务。

针对不同的内在任务，在原文的语料当中寻找相似的任务，并且将相似的任务拼接起来作为meta-train的实例，如下图所示：

![Desktop View](/assets/img/posts/2023-07-30-A-Survey-on-In-context-Learning/PICL.png)

和上篇工作太过相似了，唯一值得强调的一点，这里的语料泛化性更强，是一个大规模的通用语料库。

此外，这里利用对比学习的方法来寻来你了一个任务语义编码器，来构造各种不同的内在任务集合。将相同内在任务的两个段落视为正对，将不同的任务段落视为负对，进行对比学习。

## Inference

### Demonstration Designing

前面提到的内容是训练部分的提升，然而对于chatGPT这样的模型，其并不开源训练端口，在这样的模型上进行ICL学习是不能使用微调参数的做法的。

然而好在许多研究都表明了，ICL的性能依赖于演示(demonstration)本身，包括演示的格式以及演示的示例顺序等等，因此后文的内容会从演示组织(demonstration organization)以及演示格式(demonstration formatting)上两个方面进行介绍。

#### Demonstration Organization

所谓演示组织，其实更侧重于如何选择示例以及选择示例的顺序。

##### Demonstration Selection

**无监督方法**

1. What Makes Good In-Context Examples for GPT-3?

其方法比较符合直觉，直接选用语义上和测试样本最相似的示例来制定相关的上下文样本，原文当中特别提到了，针对“表格-文本生成”以及“开放域问答”这两个任务取得了更加显著的收益。

具体来说，选用相似的句子采用了BERT，RoBERTa或者XLNet这类语义模型/在特定任务和数据集上微调的句子编码器，根据CLS标签将句子分隔开来，同时使用简单的KNN网络来选取前k个相似的句子作为示例。

感觉操作起来比较简单，所使用的方法也比较符合直觉。

2. An Information-theoretic Approach to Prompt Engineering Without Ground Truth Labels

其方法也符合直觉，但是略微有些复杂，贴一张图来简单解释一下。

![Desktop View](/assets/img/posts/2023-07-30-A-Survey-on-In-context-Learning/MI-frame.png)

- 在训练数据上，针对每个不同的prompt模板，直接扔进语言模型，让其给出对于不同回答的概率分布（这里的概率分布包括首字母大小写的两种情况），根据大小写的概率分布加权求和，给出正确答案。
- 在prompt木板上，针对每个不同的模板后，可以根据产生的输出和对应的模板的互信息来判断，相关性越强的是更好的模板。

3. Demystifying Prompts in Language Models via Perplexity Estimation

主要的思路题目上都说了，选择prompt当中的示例数据的过程中使用困惑度作为判断指标，主要的motivation来源于，提示的性能与模型熟悉它包含的语言是相关的。

本文主要的操作流程如下所示：

- 手写一部分prompt模板信息
- 利用GPT3生成每个手动构造的模板的释义
- 回译生成的释义内容，目标是为每个任务获得大概100个提示

针对这些提示进行了实验，发现困惑度(PPL)越低，任务的性能更好。

ps:我怎么觉得这部分内容更像是demonstration formatting部分的内容，我不好说。

4. Diverse Demonstrations Improve In-context Compositional Generalization

ps:大家的文章题目都起的这么直白，一眼就懂做的是啥了。

使用了不同的演示作为ICL的输入数据，提升了输入数据的多样性。

给了一张示例图，虽然完全没看懂画的是什么东西，但是看出来更多样的meta-train数据可以取得更好的训练效果。

![Desktop View](/assets/img/posts/2023-07-30-A-Survey-on-In-context-Learning/diverse-f1.png)

本文多样性生成的一个出发点在于，预测局部结构比预测全局结构要容易很多，因此设计了一个算法来选择局部结构不同的实例；与此同时，在语义上，利用TF-IDF来计算出不同prompt的主题向量，计算其相似度来选择语义上的多样性。通过这两种方法选择多样性的数据用作prompt数据。

5. Self-Generated In-Context Learning: Leveraging Auto-regressive Language Models as a Demonstration Generator

出发点在于，像前面的文章讲述的从数据集本身选择的demonstration会造成LLMs的高度依赖，因此本文直接让LLM(其实原文说的是PLM)自己生成demonstration，最小化对外部演示的依赖。

![Desktop View](/assets/img/posts/2023-07-30-A-Survey-on-In-context-Learning/SG-ICL-f1.png)

如上图所示，先让模型自己产生demonstration，然后直接拿来拼到自己产生的prompt当中使用。思路过于简单，就不细说了。

6. Self-adaptive In-context Learning

思路很简单，先选择后排序，根据相似度等方法选择出部分候选demonstration，然后根据已有的demonstration进行排序，值得一提的是，他这里的排序使用了MDL(mininum description length)，利用压缩完的码长决定已有的demonstration的排序。但是可惜我不太会信息论，这个部分没看懂。

![Desktop View](/assets/img/posts/2023-07-30-A-Survey-on-In-context-Learning/MDL.png)

综述好长，未完待续。
