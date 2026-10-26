# OCRBench 论文精读总结

重点：论文思想、核心术语、实验结论、OCR 与 VLM 的工程应用

论文： OCRBench: On the Hidden Mystery of OCR in Large Multimodal Models

中文： OCRBench：揭示大型多模态模型中 OCR 能力的奥秘

作者： Yuliang Liu（刘昱良）等

发表： 2024 年，Science China Information Sciences

原文： [arXiv](https://arxiv.org/abs/2305.07895) ｜ [GitHub 官方代码和数据集](https://github.com/Yuliang-Liu/MultimodalOCR)

本文总结的是 OCRBench 原版，不是 OCRBench v2。

## 一、先用一分钟理解这篇论文

### 1. 核心思想

过去的 OCR 系统一般把文字检测、文字识别和信息提取分成不同任务。

例如：

传统 OCR Pipeline（处理流程）

输入图像（Image）

文字检测（Text Detection）

文字识别（Text Recognition）

信息提取（Information Extraction）

结构化结果（Structured Output）

而 VLM（Vision-Language Model，视觉语言模型）可以通过一张图片和一句自然语言指令，直接尝试完成文字识别、字段提取、文档问答等多种任务。

于是产生一个重要问题：VLM 看上去什么都能做，但它真的能把文字识别准确吗？

OCRBench 就是为了回答这个问题而设计的。

它不是提出一种新的 OCR 识别算法，而是提出一套 Benchmark（评测基准），系统地测量大型多模态模型的 OCR 能力。

### 2. 论文最重要的结论

用一句话总结：

大型多模态模型善于理解文字的含义，但理解文字，不等于能够准确地识别每一个字符。

论文发现：

- 对普通印刷文字，大型多模态模型可以有不错的表现。
- 对无意义字符、手写文字、中文、复杂公式，模型容易出错。
- 模型有时会根据语言知识猜测文字，而不完全依靠图像中的字符。
- 高分辨率视觉输入对小字、复杂文档和信息提取非常重要。
- 在严格的文字识别任务上，专用 OCR 模型仍有明显价值。

  [image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

  arXiv

  +1



## 二、OCRBench 到底测试什么？

OCRBench 原版从多种公开数据集中整理出 1,000 个经过人工检查和修正的问答样本，分成五类任务。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1



| 任务（英文与中文）                                                        | 样本数   | 测试什么             |
| ---------------------------------------------------------------- | ----- | ---------------- |
| Text Recognition（文字识别）                                           | 300   | 能否正确读出文字         |
| Scene Text-Centric VQA（场景文字视觉问答）                                 | 200   | 能否阅读场景文字并回答问题    |
| Document-Oriented VQA（文档视觉问答）                                    | 200   | 能否理解文档并回答问题      |
| KIE — Key Information Extraction（关键信息提取）                         | 200   | 能否提取指定字段         |
| HMER — Handwritten Mathematical Expression Recognition（手写数学公式识别） | 100   | 能否识别公式并转换为 LaTeX |
| Total（总计）                                                        | 1,000 | 满分 1,000 分       |

可以将这五类能力理解为三个层次：

第一层：看清楚

Text Recognition — 识别每一个字符

第二层：看明白

Scene Text VQA / Document VQA — 理解图像或文档中的文字，回答问题

第三层：提取和转换

KIE / HMER — 提取字段或将视觉内容转换成结构化表达

这里需要注意：这是一种方便记忆的能力划分，不代表论文要求模型必须经过三个处理阶段。

## 三、五类任务具体是什么？

### 1. Text Recognition — 文字识别

这是最接近传统 OCR 的任务：给模型一张包含文字的图片，让它准确输出其中的文字。

OCRBench 将这部分进一步分成六种类型，每种 50 个样本。

| 英文术语              | 中文解释  | 例子            |
| ----------------- | ----- | ------------- |
| Regular Text      | 规则文字  | 清晰、水平的印刷文字    |
| Irregular Text    | 不规则文字 | 倾斜、弯曲、透视变形的文字 |
| Artistic Text     | 艺术字体  | 广告、Logo、装饰性文字 |
| Handwritten Text  | 手写文字  | 手写英文单词        |
| Digit String      | 数字字符串 | `39280174`    |
| Non-Semantic Text | 无语义文字 | `xqztrp`      |

其中最值得关注的是 Non-Semantic Text（无语义文字）。

为什么？

因为模型识别 `APPLE` 可能很准确，但是识别 `PPAEL` 时，却可能错误地输出 `APPLE`。

这说明模型可能是在根据已有的词汇知识推断，而不是完全依赖视觉信息逐字符识别。

对于 OCR 工程，这个问题非常重要，因为证件号码、车牌、序列号、验证码等内容通常不能依靠语义猜测。

### 2. Scene Text-Centric VQA — 场景文字视觉问答

VQA = Visual Question Answering，视觉问答。

任务不是简单地识别文字，而是结合图像中的文字和问题，生成答案。

例如：

- 图片：一家商店的招牌上写着 `OPEN 9 AM – 6 PM`
- Question：What time does the store close?
- Answer：6 PM

这里不仅需要 OCR，还需要理解问题。

论文使用了 TextVQA、STVQA、OCRVQA 等相关数据集。

### 3. Document-Oriented VQA — 文档视觉问答

与 Scene Text VQA 不同，这类任务更关注文档中的复杂内容，例如发票、报告、图表、信息图。

例如：

- 输入：一张财务报表
- Question：What was the total revenue in 2023?
- Answer：$2.5 million

模型需要同时处理文字、布局、表格关系，有时还需要简单推理。

相关数据集包括：

- DocVQA：文档视觉问答
- ChartQA：图表问答
- InfographicVQA：信息图问答

区别在于：传统 OCR 主要负责把字读出来，DocVQA 还要知道哪些文字能够回答问题。

### 4. KIE — Key Information Extraction（关键信息提取）

这是与证件 OCR、发票 OCR 最相关的任务之一。

例如，输入一张发票，输出：

```
{
  "company": "ABC Store",
  "date": "2026-10-14",
  "total": "125.00"
}
```

其中需要两种不同的能力：

1. Recognition（识别）：能否识别 `125.00`？
2. Extraction（提取）：能否判断 `125.00` 是订单总金额，而不是某一商品的金额？

OCRBench 使用了 SROIE（收据）、FUNSD（表单）、POIE（产品信息）等数据集。

值得注意的是，原版 OCRBench 主要采用对字段答案提问的方式，而不是要求模型一次性输出完整的 JSON。

### 5. HMER — Handwritten Mathematical Expression Recognition（手写数学公式识别）

例如，把手写的数学公式转成 LaTeX：

图像中的表达式：

\\[ x^2+\frac{a}{b} \\]

输出：

```
x^2+\frac{a}{b}
```

难点不仅在于识别字符，还包括理解二维空间结构，例如上下标、分数、根号和括号。

论文使用 HME100K 数据集进行评估。

## 四、论文最值得记住的五个实验发现

### 发现一：Semantic Reliance — 语义依赖

这是我认为整篇论文最重要的发现。

有语义文字

# Message

熟悉的英文单词

无语义文字

# egaesMs

字母被打乱

论文发现，将正常单词中的字母顺序打乱后，受测 LMM 在 NST（Non-Semantic Text，无语义文字）上的识别准确率相对于 ST（Semantic Text，有语义文字）平均下降约 57%，而专用场景文字识别模型仅下降约 4.6%。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



这说明：

- 传统 OCR 模型通常更直接地学习字符的视觉形状。
- 多模态模型可能更强烈地利用语言先验知识（Language Prior）来生成答案。

但需要强调，这不意味着 VLM 完全不识别字符；它反映的是在论文测试条件下，语言先验对其输出有明显影响。

### 发现二：Fine-grained Perception — 细粒度视觉感知不足

论文发现，早期许多多模态模型的视觉输入分辨率只有 224×224 或 336×336。

当整页文档被缩放到这种分辨率时，小字可能已经无法清晰辨认。

例如：

原始文档

2480×3508

视觉编码

224×224

生成答案

细节可能丢失

示意图：尺寸仅用于说明缩放问题，不代表所有 VLM 都采用这一处理流程。

这也是为什么高分辨率处理（High-Resolution Processing）、图像裁剪（Image Cropping）、多尺度处理（Multi-scale Processing）对 OCR-VLM 特别重要。

论文中的 Monkey 等高分辨率模型，在文档问答和关键信息提取上表现更好。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1



### 发现三：Handwriting Recognition — 手写识别仍然困难

手写文字存在字符形状变化大、粘连、模糊等问题。

论文报告，在其手写文字对比实验中，LMM 的表现平均比专用监督模型低约 51.9%。

这里的结论针对论文测试的模型和数据集，不能直接当成 2026 年所有 VLM 的表现。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



### 发现四：Multilingual OCR — 多语言识别能力不均衡

论文发现，部分模型在英文任务上表现不错，但中文文字识别和中文视觉问答仍存在明显差距。

其原因可能包括训练数据分布、语言能力和视觉编码器对不同字符形态的支持程度。

对多国证件 OCR 来说，不能根据英文数据集的表现推断一个模型在阿拉伯文、泰文、中文等文字上也会表现良好。

### 发现五：Instruction Following — 指令遵循问题

论文举了一个很有代表性的错误：

图像中的原始日期为 `02/02/2018`，模型却输出 `2 February 2018`。

从语义上说，这两个结果可能表达相同日期，但对于要求原文转写的 OCR 任务，这不是完全相同的输出。

这揭示了 OCR 与 VLM 的一个关键区别：

OCR 有时要求忠实于原始字符（Faithful Transcription），而 VLM 往往倾向于生成规范化、符合语言习惯的答案。&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



## 五、实验结果和评测方法

### 1. 模型表现如何？

以下摘取论文 Table 3 中的部分早期模型结果，满分为 1,000 分。

OCRBench 早期模型得分对比

论文 Table 3；历史基线，不代表当前排名

0200400600800GeminiGPT-4VMonkeymPLUG-Owl2LLaVA 1.5-13B

原文 Table 3 数据。Gemini 及 GPT-4V 均指当时的模型版本。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



一个特别有意思的数据是：

| 模型     | OCRBench 总分 | HMER 得分（满分100） |
| ------ | ----------- | -------------- |
| Gemini | 659         | 8              |
| GPT-4V | 645         | 9              |
| Monkey | 514         | 0              |

早期强模型在数学公式识别上几乎无法得到高分，说明通用视觉理解能力并不能保证复杂结构的精确转写。

论文 2024 年修订版还收录了后来模型的结果，如 MiniCPM-V 2.6 达到 852 分，说明这个领域已经快速进步。因此，不能把 Table 3 当作今天的模型排名。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



### 2. OCRBench 如何计算分数？

论文为了适应生成式模型的自由文本回答，采用了比较简单的判断标准：

Ground Truth（标准答案）是否包含在模型输出中。

例如：

标准答案：ABC123

匹配

模型输出：The text is ABC123.

包含正确答案，因此通过

标准答案：ABC123

不匹配

模型输出：ABC128

字符不同，因此不通过

每个问答样本贡献一分，五种任务相加得到总分。论文也讨论了传统的 ANLS 和 Exact Matching 为什么不适合直接用于当时这种开放式的生成回答。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1



这个评分方法简单、便于比较模型，但也存在局限：

- 只要答案是输出的子字符串就可能得分，可能出现误判。
- 不能清楚衡量识别结果究竟错了几个字符。
- 不评估推理延迟、显存占用、部署成本等工程指标。
- 不能直接反映完整证件 OCR 的生产可用性。

还有一个容易混淆的细节：论文中提到的 29 个数据集属于整体研究评测范围，而 OCRBench 最终公开的 1,000 样本是从其中代表性的任务与数据集选取、整理形成的，并非对 29 个数据集全部进行等量采样。

## 六、核心术语速查表

| 英文术语                    | 中文                    | 直观理解                            |
| ----------------------- | --------------------- | ------------------------------- |
| Benchmark               | 评测基准                  | 一套统一的考试题                        |
| Baseline                | 基线模型/基准结果             | 后续模型比较的参照                       |
| LMM                     | 大型多模态模型               | 能处理视觉和语言等多模态输入的大模型              |
| VLM                     | 视觉语言模型                | 将图像与语言结合的模型                     |
| Zero-shot               | 零样本                   | 不针对当前任务提供训练示例，直接测试              |
| Ground Truth (GT)       | 标准答案/真实标注             | 用来判断预测是否正确的答案                   |
| Semantic Reliance       | 语义依赖                  | 根据词义和上下文辅助甚至替代字符判断              |
| Language Prior          | 语言先验                  | 模型提前学到的语言统计规律                   |
| Fine-grained Perception | 细粒度感知                 | 区分小字、细微笔画和相似字符                  |
| KIE                     | 关键信息提取                | 从图像中定位并提取目标字段                   |
| HMER                    | 手写数学表达式识别             | 将手写公式转为结构化表达                    |
| ANLS                    | 平均归一化 Levenshtein 相似度 | 使用编辑距离评价文本预测                    |
| CER                     | 字符错误率                 | 衡量识别结果在字符层面的错误程度                |
| WER                     | 词错误率                  | 衡量识别结果在单词层面的错误程度                |
| SOTA                    | 当前最先进水平               | State of the Art，特定时期、数据集上的优秀结果 |
| Ablation Study          | 消融实验                  | 去掉或改变某个模块，分析其贡献                 |

## 七、从 OCR 工程师角度，这篇论文有什么实际价值？

结合传统证件 OCR、文字检测识别和字段提取的工作场景，我认为值得带走三点。

### 1. OCR 的识别能力和理解能力必须分开评估

例如一张护照图片包含：

- Name：ZHANG XIN
- Passport No.：E12345678
- Date of Birth：12 OCT 1987

模型可能知道哪个字段是护照号码，但是不一定能保证每一个字母和数字都识别正确。

因此，建议区分：

| 目标      | 应用评价指标                        |
| ------- | ----------------------------- |
| 字符准确性   | CER、Exact Match               |
| 整个字段准确性 | Field Exact Match             |
| 字段提取完整性 | Field Recall                  |
| 整份证件正确率 | Document Exact Match          |
| 性能      | Latency、Throughput、GPU Memory |

这些是工程评估建议，不是 OCRBench 原版已经覆盖的全部指标。

### 2. 不应该简单地把传统 OCR Pipeline 全部替换成 VLM

更合理的是根据任务特点选择。

专用 OCR 更适合的方向

固定格式证件、MRZ、精确数字字符串、低延迟和高吞吐的识别任务。

VLM 更值得发挥作用的方向

开放式文档、复杂布局、自然语言问答、长尾字段提取以及需要上下文理解的任务。

Hybrid OCR-VLM（混合方案）

专用 OCR 负责高保真文字转写，VLM 负责复杂文档理解和语义信息提取，最后再做规则校验。

这不意味着 VLM 在任何情况下都比专用 OCR 的识别精度低。最终仍需针对实际数据、模型版本、成本和业务约束测试。

### 3. OCRBench 原版已经不能代表完整的现代 OCR-VLM 评测

这篇论文的重要价值是首次系统揭示了很多早期 LMM 的文字识别弱点。

不过，原版样本量只有 1,000，任务覆盖、文本定位能力和评分细致程度都有局限。

后续论文 OCRBench v2: An Improved Benchmark for Evaluating Large Multimodal Models on Visual Text Localization and Reasoning 扩展到了 10,000 个经过人工验证的问答样本，覆盖 31 类场景，更强调文字定位、细粒度感知、复杂元素解析和逻辑推理。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1



## 八、面试应该怎么讲这篇论文？

建议记住下面这段英文，能够概括主要观点，不需要背复杂实验数字。

> OCRBench is a benchmark designed to evaluate the OCR capabilities of large multimodal models. It covers five major tasks, including text recognition, visual question answering, key information extraction, and handwritten mathematical expression recognition.
>
> One important finding is that multimodal models rely heavily on semantic information. They may recognize meaningful words correctly but struggle with random character sequences.
>
> This suggests that traditional OCR models are still valuable for high-precision text recognition, while VLMs offer advantages in document understanding and complex information extraction.

## 九、最后总结：这篇论文应该记住什么？

| 问题                | 核心答案                                 |
| ----------------- | ------------------------------------ |
| What：它是什么？        | 一套评估多模态模型 OCR 能力的 Benchmark          |
| Why：为什么提出？        | 大模型能看懂文字，但未必能准确读出每一个字符               |
| How：怎么评估？         | 5 大任务、1,000 个问答样本，统一打分               |
| Findings：最重要发现？   | 语义依赖、细粒度感知不足、手写和复杂公式识别困难             |
| Limitation：局限性？   | 样本少、评分较粗糙、不覆盖完整工程指标                  |
| Engineering：如何应用？ | 将精确字符识别与文档理解分开评估，按场景选择 OCR、VLM 或混合方案 |

我的评价： 这篇论文值得读，但不必花很多时间研究实验表格。它主要是一篇评测论文，而不是提出新的模型架构。作为有传统 OCR 经验、正在补充 VLM 知识的算法工程师，最有价值的是理解 Semantic Reliance、Fine-grained Perception 和 KIE 这三个问题。

如果继续学习，我建议下一篇读 OCRBench v2，再进一步看实际端到端文档解析模型的论文。这样能从“如何评价 VLM 的 OCR 能力”，自然过渡到“如何构建和优化 VLM-OCR 系统”。



