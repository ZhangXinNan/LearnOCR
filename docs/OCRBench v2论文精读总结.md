# OCRBench v2 论文精读总结

核心：从文字识别，扩展到文字定位、结构解析、信息提取和复杂推理

论文名称： OCRBench v2: An Improved Benchmark for Evaluating Large Multimodal Models on Visual Text Localization and Reasoning

中文： OCRBench v2：一个用于评估大型多模态模型视觉文字定位与推理能力的改进基准

作者： Ling Fu 等，华中科技大学等机构

发表： NeurIPS 2025，Datasets and Benchmarks Track（数据集与基准评测方向）

论文资源： [arXiv 论文](https://arxiv.org/abs/2501.00321) ｜ [GitHub 官方代码](https://github.com/Yuliang-Liu/MultimodalOCR) ｜ [官方项目与排行榜](https://99franklin.github.io/ocrbench_v2/)

## 一、先用一分钟理解这篇论文

### 1. 为什么需要 OCRBench v2？

上一版 OCRBench 主要回答：

VLM 能不能准确识别图片里的文字，并根据文字回答问题？

随着 VLM 的发展，简单文字识别已经不够难了。

例如，论文指出，Qwen2.5-VL 在当时的 DocVQA 数据集上达到 96.4%，在原版 OCRBench 上达到 88.8%。

但这是否代表 VLM 已经能够替代传统 OCR 系统？

不一定。

因为一个模型可能：

- 能读出某个单词，却无法准确给出单词的 Bounding Box（边界框）。
- 能提取一张发票的金额，却无法正确对应商品、数量和价格。
- 能读出表格中的文字，却无法恢复表格的行列结构。
- 能理解一道数学题，却无法准确计算答案。

OCRBench v2 的核心思想：不能只考模型“能否读出文字”，还要考它“能否定位、组织、关联、理解和推理文字”。&#x20;

[image](https://www.google.com/s2/favicons?domain=https://proceedings.nips.cc\&sz=32)

NIPS 会议论文集

+1



### 2. 原版与 v2 的区别

| 对比项目    | OCRBench v1 | OCRBench v2         |
| ------- | ----------- | ------------------- |
| 核心能力分类  | 5 类         | 8 类                 |
| 具体任务    | 5 类核心任务     | 23 项                |
| 场景覆盖    | 约 14 类      | 31 类                |
| 评测问答数   | 1,000       | 10,000              |
| 图片数量    | 约 900       | 约 9,500             |
| 评分      | 相对简单的答案匹配   | 按任务采用不同评价指标         |
| 重点      | 文字识别与文字理解   | 文字定位、关系提取、结构解析、复杂推理 |
| 独立私有测试集 | 未作为核心设计     | 额外 1,500 张图片        |

依据论文 Table 1 与数据集设计章节整理。这里的“31 类”是应用场景（Scenarios），不是 31 种任务。

[image](https://www.google.com/s2/favicons?domain=https://proceedings.neurips.cc\&sz=32)

NeurIPS 网址

+1

### 3. 论文最重要的结论

论文评估了多个当时先进的多模态模型，发现大多数模型的 OCRBench v2 平均分低于 50/100。

这不是说模型只有不到 50% 的字符识别准确率，而是指它们在多项复杂 OCR 相关能力上的综合表现仍然不足。

特别是：

1. Text Localization（文字定位）：难以精确找到文字的位置。
2. Fine-grained Perception（细粒度感知）：容易看错小字符、密集字符。
3. Layout Perception（布局感知）：对复杂文档空间结构理解不充分。
4. Complex Element Parsing（复杂元素解析）：表格、图表、公式等结构难以准确还原。
5. Logical Reasoning（逻辑推理）：即使识别了文字，也未必能完成正确推理。

这五个方面是论文总结的主要不足，也是阅读本文最值得抓住的内容。

[image](https://www.google.com/s2/favicons?domain=https://proceedings.nips.cc\&sz=32)

NIPS 会议论文集

+1



## 二、OCRBench v2 测试哪八大能力？

OCRBench v2 的任务分类图：8 大能力、23 项具体任务。

[image](https://www.google.com/s2/favicons?domain=https://proceedings.neurips.cc\&sz=32)

NeurIPS 网址

这部分是整篇论文最值得理解的内容。

| 8 大能力（英文/中文）                        | 具体任务                                                                         | 核心问题            |
| ----------------------------------- | ---------------------------------------------------------------------------- | --------------- |
| 1. Text Recognition 文字识别            | Text Recognition、Fine-grained Text Recognition、Full-page OCR                 | 能读对吗？           |
| 2. Text Referring 文字指代定位            | Text Grounding、VQA with Position                                             | 能找到指定文字的位置吗？    |
| 3. Text Spotting 文字检测与识别            | Text Spotting                                                                | 能一次找出所有文字及其位置吗？ |
| 4. Relation Extraction 关系提取         | KIE、Key Information Mapping、Handwritten Content Extraction                   | 能找对字段及对应关系吗？    |
| 5. Element Parsing 元素解析             | Table、Chart、Document Parsing、Formula Recognition                             | 能还原文档结构吗？       |
| 6. Mathematical Calculation 数学计算    | Math QA、Text Counting                                                        | 能数对、算对吗？        |
| 7. Visual Text Understanding 视觉文字理解 | Document Classification、Diagram QA、Cognition VQA                             | 能理解文字与图像吗？      |
| 8. Knowledge Reasoning 知识推理         | Science QA、Reasoning VQA、APP Agent、ASCII Art Classification、Text Translation | 能综合知识进行推理吗？     |

下面重点解释最有工程价值的部分。

### 1. Text Recognition — 文字识别

v2 不只是识别单词，而是增加了更复杂的文字类型和场景。

例如：

- Fine-grained Text Recognition（细粒度文字识别）：给定一块指定区域，识别其中很小或很密集的文字。
- Full-page OCR（整页 OCR）：把整张文档内的文字全部识别出来，并尽量保持正确阅读顺序。
- Occluded Text（遮挡文字）：字符被部分遮挡。
- Dot-matrix Text（点阵文字）：字符由离散点组成，常见于票据、工业喷码等。

这里的核心是：单行清晰文字识别准确，不代表整页复杂文档能够准确还原。

### 2. Text Referring — 文字指代定位

这是 v2 的重要新增能力。

假设图片中有很多文字：

示意：一张包含多个字段的图像

NAME

ZHANG XIN

PASSPORT NO.

E12345678

DATE OF BIRTH

12 OCT 1987

问题：Where is the text "E12345678"?

输出：指定文字的 Bounding Box（边界框）

相关术语：

Text Grounding（文字定位）：给定文字，找出它在图片中的位置。

VQA with Position（带位置的视觉问答）：不仅回答问题，还要给出答案所在的准确坐标。

论文统一采用 0–1000 的归一化坐标范围来表示位置，便于处理不同尺寸的图片。

[image](https://www.google.com/s2/favicons?domain=https://proceedings.neurips.cc\&sz=32)

NeurIPS 网址



### 3. Text Spotting — 文字检测与识别

这部分很容易与 Text Referring 混淆。

| 对比        | Text Referring | Text Spotting |
| --------- | -------------- | ------------- |
| 输入        | 图片 + 指定文字或问题   | 整张图片          |
| 输出        | 指定文字的坐标        | 所有文字及其坐标      |
| 直观理解      | 指哪找哪           | 全部找出并读出       |
| 传统 OCR 对应 | 文字定位/目标定位      | 检测 + 识别       |

例如，Text Spotting 要求输出：

```
[
  {
    "text": "NAME",
    "bbox": [100, 150, 250, 190]
  },
  {
    "text": "ZHANG XIN",
    "bbox": [300, 150, 600, 190]
  }
]
```

示意输出，坐标并非论文真实样本。

可以把它理解为传统的 DBNet + CRNN/SVTR 等检测识别 Pipeline 的端到端任务。

这正是 OCRBench v2 的一个重要发现：部分 VLM 已经很擅长识别文字，但同时输出准确文字和准确坐标仍然困难。

### 4. Relation Extraction — 关系提取

不仅要识别文字，还要找出文字之间的关系。

例如一张收据：

| Item   | Qty | Price |
| ------ | --- | ----- |
| Coffee | 2   | $10   |
| Cake   | 1   | $7    |

如果只做 OCR，系统可能输出：

`Coffee, 2, 10, Cake, 1, 7`

但 Relation Extraction 要进一步确定：

`Coffee → Qty 2 → Price 10`

其中最重要的两个术语：

KIE（Key Information Extraction，关键信息提取）：给定字段名，从图像中提取对应字段值。

Key Information Mapping（关键信息映射）：在若干字段名和字段值之间建立正确匹配关系。

它们的区别是：KIE 需要从图像中找到值；Mapping 更强调已经提供的键和值如何正确配对。

这在发票、身份证、护照、银行流水中都很常见。

### 5. Element Parsing — 复杂元素解析

这是文档 AI 的核心能力之一。

| 任务                  | 中文解释 | 典型输出            |
| ------------------- | ---- | --------------- |
| Table Parsing       | 表格解析 | HTML / Markdown |
| Chart Parsing       | 图表解析 | JSON / Dict     |
| Document Parsing    | 文档解析 | Markdown        |
| Formula Recognition | 公式识别 | LaTeX           |

例如，一张图片中有下面的公式：

\\[ E = mc^2 \\]

模型除了能识别 `E`、`m`、`c` 和 `2`，还需要理解数字 2 是上标。

再例如表格识别，模型不能只是读出每个单元格的文字，还必须判断行、列以及合并单元格的关系。

识别内容正确，并不意味着结构解析正确。

这也是为什么现代文档 AI 会把 Layout Analysis（版面分析）与 Element Parsing（元素解析）视为重要能力。

### 6–8. 数学计算、视觉理解与知识推理

这三类任务属于更高层次的能力。

| 任务                            | 举例               | 必需的能力      |
| ----------------------------- | ---------------- | ---------- |
| Text Counting（文字计数）           | 图片中有多少个单词？       | 视觉计数       |
| Math QA（数学问答）                 | 读取几何题并计算角度       | OCR + 数学推理 |
| Document Classification（文档分类） | 判断是发票、简历还是报告     | 文档语义理解     |
| Diagram QA（图示问答）              | 理解流程图中的连接关系      | 空间关系理解     |
| Cognition VQA（认知型视觉问答）        | 图表中蓝色曲线是否出现数值30？ | 读取并理解图像    |
| Reasoning VQA（推理型视觉问答）        | 根据数据判断趋势         | 逻辑推理       |
| APP Agent（应用界面智能体）            | 根据截图识别可用操作       | UI 理解与常识   |

这里我认为一个很重要的区别是：

Cognition VQA：答案通常可以直接从图像中找到。

Reasoning VQA：答案不直接存在，需要结合信息进行推理。

例如：

图像文字：`Coffee $5, Cake $7`

- Cognition VQA：How much is the cake? → $7
- Reasoning VQA：How much for two coffees and one cake? → $17

## 三、论文最重要的五个实验发现

### 发现一：VLM 的文字识别已经很强，但复杂文字仍然困难

论文附录 Table 8 对比了一些专用 OCR 模型和 VLM。

| 模型            | 文字识别得分 |
| ------------- | ------ |
| CRNN          | 38.1   |
| ABINet        | 62.4   |
| SVTR          | 57.8   |
| Qwen2.5-VL-7B | 73.0   |
| InternVL3-14B | 71.1   |
| GPT-4o        | 74.1   |

OCRBench v2 附录 Table 8，特定测试集上的历史结果，不代表所有 OCR 场景。

[image](https://www.google.com/s2/favicons?domain=https://proceedings.neurips.cc\&sz=32)

NeurIPS 网址

这其实纠正了一个容易产生的误解：

不能再笼统地认为传统 OCR 的识别精度一定超过 VLM。

在论文选取的这些文字识别样本上，部分 VLM 已经超过了所测试的传统识别模型。

但是对低频文字（Low-frequency Text）仍然存在明显差距。

InternVL3-14B 不同文字类型的识别表现

High-frequency Text

# 79.1%

高频常见文字

Low-frequency Text

# 46.7%

低频特殊文字

论文 Table 6。低频类别包括 CAPTCHA、点阵文字、部分遮挡文字等。

[image](https://www.google.com/s2/favicons?domain=https://proceedings.neurips.cc\&sz=32)

NeurIPS 网址

### 发现二：文字定位能力远落后于内容识别能力

这是我认为全文最有价值的实验之一。

在论文 Table 7 的 VQA with Position 测试中：

Content Accuracy

# 78.3%

内容回答准确率

IoU Accuracy

# 12.9%

位置准确率

InternVL3-14B，论文 Table 7。两个指标的评价方法不同，不能直接理解为同一个任务的字符正确率与检测召回率。

[image](https://www.google.com/s2/favicons?domain=https://proceedings.neurips.cc\&sz=32)

NeurIPS 网址

意思是，模型能够回答正确的文字内容，但不一定知道对应文字的精确区域。

对 OCR 工程而言，这可能是问题，因为很多业务不仅需要字段值，还需要提供位置用于审核、标注和可视化。

### 发现三：复杂布局影响非常大

论文测试了文档旋转情况。

InternVL3-14B 在 DocVQA 上：

原始图像

# 90.9%

旋转90°

# 35.2%

准确率下降 55.7 个百分点。

[image](https://www.google.com/s2/favicons?domain=https://proceedings.neurips.cc\&sz=32)

NeurIPS 网址

这说明，即使 VLM 能够处理任意图片，Image Orientation（图像方向）、Reading Order（阅读顺序）和 Layout Perception（布局感知）仍然重要。

实际工程中，自动旋转校正、文本区域裁剪、版面分析仍可能提升整体效果。

### 发现四：识别文字不等于能够还原所有文字的位置和结构

论文附录 Table 9 比较了传统 Text Spotting 方法与 VLM：

| 模型             | Text Spotting F1 |
| -------------- | ---------------- |
| ABCNetV2       | 44.2             |
| TESTR          | 51.8             |
| Qwen2.5-VL-7B  | 1.2              |
| InternVL3-14B  | 11.2             |
| Gemini 1.5 Pro | 13.5             |

论文附录 Table 9。专用模型采用特定数据微调，因此不是完全同训练条件的公平对比，但仍能反映任务适配差异。

[image](https://www.google.com/s2/favicons?domain=https://proceedings.neurips.cc\&sz=32)

NeurIPS 网址

你会发现一个看似矛盾的现象：

VLM 的 Text Recognition 已经很强，但 Text Spotting 却很差。

其实并不矛盾。

因为 Text Spotting 要求同时解决文字检测、识别、精确坐标输出和多目标组织问题。

这些结果说明模型能力是分任务的，不能因为识别准确率高，就认为它能完全替代检测识别 Pipeline。

### 发现五：推理能力不稳定

论文还考察了常识推理、视觉文字理解、模式识别、数学计算和专家知识。

例如 InternVL3-14B 在私有测试集上的人工评估结果：

| 推理类型                              | 准确率   |
| --------------------------------- | ----- |
| Common Sense Reasoning（常识推理）      | 72.9% |
| Visual-text Understanding（视觉文字理解） | 83.0% |
| Pattern Recognition（模式识别）         | 69.2% |
| Calculation（计算）                   | 56.5% |
| Expert Knowledge（专家知识）            | 71.8% |

计算任务明显更困难。

[image](https://www.google.com/s2/favicons?domain=https://proceedings.neurips.cc\&sz=32)

NeurIPS 网址



这说明，大模型即使能够从图像中正确抽取数字，也不应该直接假设它能够可靠地完成全部计算。

例如发票金额、财务统计等场景，更合理的是由程序执行确定性计算和规则校验。

## 四、OCRBench v2 如何评分？

这是相对于 v1 最重要的技术变化之一。

原版主要使用相对宽松的答案包含判断。

但如果现在要评估 Bounding Box、表格结构、键值对、长文档等，就不能简单地判断某个字符串是否出现在输出中。

因此 v2 设计了六类评测方法。

| 评测类别                | 主要指标                         | 直观含义          |
| ------------------- | ---------------------------- | ------------- |
| Parsing（结构解析）       | TEDS                         | 输出结构与真实结构有多相似 |
| Localization（定位）    | IoU                          | 预测框与真实框有多大重叠  |
| Extraction（信息提取）    | F1                           | 提取的键值对有多少正确   |
| Long Reading（长文本识别） | BLEU、METEOR、F1、Edit Distance | 长文本与标准答案有多接近  |
| Counting（计数）        | L1 Distance                  | 数量预测误差有多大     |
| Basic VQA（普通问答）     | Exact Match、ANLS、答案包含判断      | 问答结果是否正确      |

[image](https://www.google.com/s2/favicons?domain=https://proceedings.neurips.cc\&sz=32)

NeurIPS 网址

+1

其中最值得掌握三个术语。

### 1. IoU — Intersection over Union（交并比）

\\[ IoU=\frac{\text{Intersection Area}}{\text{Union Area}} \\]

就是预测框和真实框的交集面积除以并集面积。

越接近 1，位置越准确。

### 2. F1 Score — F1 分数

\\[ F1=\frac{2PR}{P+R} \\]

其中：

- Precision（精确率）：提取出的结果有多少正确。
- Recall（召回率）：应该提取的结果有多少被找到了。

在 KIE 中，可以按正确匹配的 Key-Value（键值对）计算 F1。

### 3. TEDS — Tree-Edit-Distance-based Similarity（基于树编辑距离的相似度）

这是一种衡量结构化结果相似度的方法。

例如：

真实表格是 3 行 × 4 列，而模型生成 4 行 × 3 列。

即使文字基本识别正确，行列结构仍然有问题，因此 TEDS 会受到影响。

可以理解为：

Edit Distance 主要比较字符串，TEDS 更关注树形结构是否一致。

例如，HTML 表格的行列嵌套关系就可以使用树结构表示。

## 五、OCRBench v2 的实验结果该怎么看？

以下是论文中部分模型在英文公开测试集上的综合分数。

OCRBench v2 英文测试集表现

论文 Table 2，满分100；历史实验结果

0255075100InternVL3-14BGemini-ProOvis2-8BQwen2.5-VL-7BGPT-4o

论文 Table 2，Average 是不同能力得分的平均值，不是整个数据集的逐字符正确率。模型结果受具体版本和测试配置影响。

[image](https://www.google.com/s2/favicons?domain=https://proceedings.neurips.cc\&sz=32)

NeurIPS 网址

+1

这里还有两个重要细节。

第一，英文和中文是分别评测的。 在论文公开数据中，英文部分包含全部 8 类能力，中文部分报告了其中 5 类。因此不能简单地把中英文平均分当作完全相同组成的测试结果。

第二，以上都是论文发表时期的结果。 到 2026 年，官方排行榜已经包含更多的新模型，不能把这些历史结果理解成当前的性能排名。

[image](https://www.google.com/s2/favicons?domain=https://99franklin.github.io\&sz=32)

99franklin.github.io



## 六、从证件 OCR、文档 AI 的工程角度，该如何理解？

我认为，这篇论文对实际系统设计的价值，比排行榜本身更大。

### 1. 传统 OCR 和 VLM 不是简单的替代关系

可以把 OCR 系统理解为四类相互关联但可以独立评估的能力：

Recognition — 识别

字符、单词、文本行能否准确识别

Localization — 定位

文字在哪里，对应什么坐标

Structure — 结构

字段如何关联，表格如何组织

Reasoning — 推理

能否理解文字并完成业务判断

这是便于理解的能力划分，不表示实际系统必须严格按上述顺序执行。

传统 OCR Pipeline 和 VLM 都可能覆盖其中一些或全部能力，但各自有不同的优势和限制。

### 2. 证件 OCR 应该重点评估哪些能力？

对于护照、身份证、驾照和其他多国证件，我会优先关注下面这些指标。

| 能力   | 对应测试                 | 工程重要性 |
| ---- | -------------------- | ----- |
| 字符识别 | Exact Match、CER      | 极高    |
| 文字定位 | IoU、Recall           | 高     |
| 字段提取 | Field F1、Exact Match | 极高    |
| 字段关联 | Key-Value Matching   | 高     |
| 版面理解 | Layout Analysis      | 中高    |
| 推理   | Rule Validation      | 视业务需要 |

比如护照 OCR 中，MRZ（Machine Readable Zone，机器可读区）尤其强调精确转写。

模型若把 `O` 和 `0` 混淆，或者自动纠正姓名，哪怕语义上看起来合理，业务上也可能无法接受。

所以，一个 VLM 在 OCRBench v2 上的综合分很高，不代表它一定适合高精度证件 OCR。

### 3. 一种值得考虑的 Hybrid OCR-VLM 架构

Image Input — 输入图像

Traditional OCR

文字检测 + 识别 + 坐标

VLM

文档理解 + 字段提取

Result Fusion — 结果融合

Validation — 规则校验

格式检查、MRZ 校验位、字段一致性检查等

Structured Output — 结构化输出

这只是一个可选的工程架构，不是论文提出的方法，也不保证一定优于端到端 VLM。

实际选择时，还需要考虑 Latency（延迟）、Throughput（吞吐量）、GPU Memory（显存）、Inference Cost（推理成本）以及人工审核成本。

## 七、论文的局限性

OCRBench v2 比原版全面，但仍不是完整的工业 OCR 评价体系。

| 局限                                | 解释                            |
| --------------------------------- | ----------------------------- |
| Single-image Focus（单图为主）          | 主要评估单张图片，不重点研究跨页关联和长文档检索      |
| Limited Language Coverage（语言覆盖有限） | 以中英文为主，不能代表其他文字系统的效果          |
| Benchmark vs Production（评测与生产差异）  | 未系统评价部署延迟、吞吐、稳定性、总体成本         |
| Evaluation Sensitivity（评分敏感性）     | JSON、Markdown、坐标格式和评价规则可能影响分数 |
| Benchmark Contamination（基准数据污染）   | 公开数据可能出现在模型训练数据中，故额外构建私有测试集   |

论文设计的 1,500 张私有图片，正是为了缓解公开数据可能被模型训练过程覆盖的问题。

[image](https://www.google.com/s2/favicons?domain=https://proceedings.neurips.cc\&sz=32)

NeurIPS 网址

+1



## 八、核心术语速查表

| 英文                      | 中文                    | 记忆方式               |
| ----------------------- | --------------------- | ------------------ |
| Text Recognition        | 文字识别                  | 读对文字               |
| Fine-grained Perception | 细粒度感知                 | 看清细小字符             |
| Text Referring          | 文字指代定位                | 指哪找哪               |
| Text Grounding          | 文字定位                  | 给定文字找坐标            |
| Text Spotting           | 文字检测与识别               | 全部找出并读出            |
| Bounding Box / BBox     | 边界框                   | 文字位置               |
| Relation Extraction     | 关系提取                  | 找对字段关系             |
| KIE                     | 关键信息提取                | 提取字段               |
| Layout Perception       | 布局感知                  | 理解空间排列             |
| Element Parsing         | 元素解析                  | 还原表格、公式等结构         |
| TEDS                    | 基于树编辑距离的相似度           | 比较结构               |
| IoU                     | 交并比                   | 比较两个框              |
| ANLS                    | 平均归一化 Levenshtein 相似度 | 比较文本答案             |
| Low-frequency Text      | 低频文字                  | 不常见、难识别的文字         |
| Structured Output       | 结构化输出                 | JSON、HTML、Markdown |
| Private Test Set        | 私有测试集                 | 独立验证模型泛化能力         |
| Benchmark Saturation    | 评测基准趋于饱和              | 模型得分普遍很高，测试不够难     |

## 九、面试如何介绍 OCRBench v2？

下面这段英文可以用于 OCR/VLM 面试，重点不是死记模型得分，而是讲清这篇论文对工程设计的启发。

> OCRBench v2 is an improved benchmark for evaluating the OCR capabilities of large multimodal models.
>
> Compared with the original OCRBench, it covers more challenging tasks, including text localization, text spotting, key information extraction, document parsing, and logical reasoning.
>
> One important finding is that modern VLMs perform well on basic text recognition, but still struggle with precise text localization and complex document structures.
>
> This suggests that VLMs cannot always replace traditional OCR pipelines, especially when accurate character-level recognition and bounding boxes are required.
>
> In practical systems, a hybrid approach combining OCR and VLMs may be a good solution.

## 十、最后总结：这篇论文最值得记住什么？

| 问题                | 核心答案                            |
| ----------------- | ------------------------------- |
| What：是什么？         | 一套更全面、更困难的 OCR-VLM Benchmark    |
| Why：为什么？          | 原版过于简单，不能充分发现 VLM 的能力短板         |
| How：如何评价？         | 8 大能力、23 项任务、31 类场景、10,000 个 QA |
| Key Finding：最大结论？ | 文字识别已经很强，精准定位、结构解析和推理仍存在挑战      |
| Engineering：如何应用？ | 不要只看识别准确率，要分别评估检测、定位、提取、结构和推理   |
| Limitation：不足？    | 单图任务为主、语言覆盖有限、缺少工程性能评价          |

我的评价：OCRBench v2 比原版更值得你仔细阅读。

因为原版主要说明了 VLM 在文字识别中的局限，而 v2 更进一步展示了：VLM 的文字识别能力已经取得很大进步，真正需要继续解决的还有精准文字定位、空间布局理解、结构化输出与复杂推理。

如果学习时间有限，我建议重点阅读原论文中的 Figure 3（23 项任务）、Table 2/3（模型能力对比）、Section 5.3（五个实验发现）、Appendix A.1（与传统 OCR 模型对比）。

这四部分基本涵盖了论文对 OCR 算法工程师最重要的价值。


