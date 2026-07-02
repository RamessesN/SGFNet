<div align="center">
    <h1> SGFNet Feedback </h1>
</div>

## Associate Editor Comments:

Associate Editor \
REQUIRED: Comments to the Author: \
This letter present a semantic-guided fusion network for multi-source remote sensing image classification. However, several critical issues need to be addressed.

副编辑 \
作者审阅要求： \
本文提出了一种基于语义引导的融合网络，用于多源遥感图像分类。然而，仍需解决若干关键问题。

---

## Reviewer(s) Comments:

### Reviewer: 1

Comments to the Author \
This paper proposes a Semantic-Guided Fusion Network (SGFNet) for multi-source remote sensing image classification. Experimental results show that SGFNet achieves competitive classification performance compared with several existing methods. However, the manuscript still has several weaknesses that should be carefully addressed:

致作者的评论： \
本文提出了一种语义引导融合网络 (SGFNet) 用于多源遥感图像分类。实验结果表明，SGFNet 在分类性能上与现有多种方法相比具有竞争力。然而，该论文仍存在若干需谨慎改进的不足之处：

1. The manuscript does not clearly explain how the proposed SMCB differs substantially from existing dynamic convolution or semantic-aware filtering methods. Please provide in-depth description to highlight the technical contribution of the proposed method.

    该手稿并未清晰说明所提出的 SMCB 与现有动态卷积或语义感知过滤方法在哪些方面存在显著差异。请提供深入的描述，以突出所提方法的技术贡献。

2. Please explain why frequency-domain modulation is more robust to misalignment and provide controlled experiments with different degrees of artificial spatial shifts to verify this claim.

    请解释为何频域调制对偏移更鲁棒，并通过不同程度的人工空间位移进行控制实验，以验证这一说法。

3. Eq. 11 and 12 appear to place the value features inside the Softmax operation, which is inconsistent with the standard attention formulation. Please check carefully or provide more descriptions.

    公式 11 和 12 似乎将值特征置于 Softmax 操作内部，这与标准的注意力建模不一致。请仔细检查或提供更多信息。

4. The paper states that the query and key features are multiplied to obtain an affinity map, and then a dynamic kernel is generated. However, the exact dimensions of the affinity map, the reshaping operation, and the dynamic kernel are not sufficiently clear. Please add more discussions.

    论文指出，通过将查询和键特征相乘来生成亲和力图，然后生成动态核。然而，亲和力图的精确维度、重塑操作以及动态核的具体细节尚不明确，请补充更多讨论内容。

5. The comparison should be expanded to include more recent and stronger baselines.

    比较范围应扩展至包含更近期且更强的基准线。

6. Although SGFNet achieves the best overall accuracy, it does not consistently outperform all baselines for every class. This indicates that the proposed method may still struggle with some spectrally ambiguous or minority classes. The authors should provide a deeper analysis of these failure cases.

    尽管 SGFNet 在整体准确率上表现最佳，但它并非对每个类别都始终优于所有基线。这表明该方法在某些光谱模糊或少数类别的任务中仍可能面临挑战。作者应对此类失败案例进行更深入的分析。

---

### Reviewer: 2

Comments to the Author \
This letter proposes SGFNet, a semantic-guided fusion network for multi-source remote sensing image classification, which consists of two key modules: the Semantic Mixing Convolution Block (SMCB) for adaptive semantic contextual modeling and the Frequency Modulated Fusion Block (FMFB) for frequency-domain cross-modal interaction to alleviate slight spatial misalignment. The motivation of addressing insufficient semantic contextual modeling and unreliable fusion caused by spatial misalignment is well-justified, and the technical design is clearly presented. However, several critical issues need to be addressed before acceptance.

致作者的评论： \
本文提出 SGFNet，一种用于多源遥感图像分类的语义引导融合网络，包含两个关键模块：用于自适应语义上下文建模的语义混合卷积块 (SMCB)，以及用于频域跨模态交互以缓解微小空间错位的频率调制融合块 (FMFB)。针对因空间错位导致的语义上下文建模不足和融合不可靠问题的动机具有充分依据，且技术设计清晰明确。然而，在被接受之前，仍需解决若干关键问题。

1.  Has the author considered the issue of class imbalance? From the experimental results, such a problem indeed exists. What are the underlying causes of this issue?

    作者是否考虑过类别不平衡的问题？根据实验结果，这一问题确实存在。该问题的根本原因是什么？

2. How does SGFNet address the issue of spatial misalignment in multi-source data?

    SGFNet 如何解决多源数据中的空间错位问题？

3. The paper mentions that SMCB can better capture long-range semantic dependencies compared to traditional convolution, and how this can be explained through physical meaning.

    论文提到，SMCB 相比传统的卷积能够更好地捕捉长距离语义依赖，并通过物理意义来解释这一现象。

4. What impact do different convolution kernel sizes in the SMCB module have on the overall detection performance?

    SMCB 模块中不同卷积核大小对整体检测性能有何影响？

---

### Reviewer: 3

Comments to the Author \
The authors present Semantic-Guided Fusion Network for multi-source remote sensing image classification. The proposed method aims to strengthen semantic contextual feature learning by introducing the Semantic Mixing Convolution Block (SMCB), while the Frequency Modulated Fusion Block (FMFB) is designed to enhance cross-modal feature interaction and reduce the adverse effects of slight spatial misalignment. However,  more convincing analysis and additional experiments are needed to support the claims:

致作者的评论： \
作者提出了一种基于语义引导的融合网络，用于多源遥感图像分类。该方法通过引入语义混合卷积块（SMCB）来加强语义上下文特征的学习，同时设计频率调制融合块（FMFB）以增强跨模态特征交互，并减少微小空间错位带来的负面影响。然而，仍需更可信的分析和进一步实验来支持上述观点。

1. SMCB involves dynamic kernel generation and semantic affinity computation, while FMFB introduces attention-like operations. These operations may increase computational cost. Please add corresponding efficiency analysis.

    SMCB 包含动态内核生成和语义亲和力计算，而 FMFB 引入了类似注意力的操作。这些操作可能会增加计算开销。请补充相应的效率分析。

2. Some parameters, such as the dynamic kernel size in SMCB, the channel dimension is not analyzed or introduced. Please add more sensitivity analysis or descriptions.

    某些参数，例如 SMCB 中的动态核大小，其通道维度未进行分析或引入。请补充更多的敏感性分析或说明。

3. Although SGFNet improves the overall performance, the classification accuracy of difficult categories remains unsatisfactory. The authors should discuss why these classes are still poorly classified.

    尽管 SGFNet 提升了整体性能，但困难类别的分类准确率仍不尽如人意。作者应探讨为何这些类别仍然分类不佳。

4. The classification maps of Augsburg are not provided. Please add more visualized classification results.

    奥格斯堡的分类地图未提供，请补充更多可视化分类结果。

5. The relationship between DCT-domain modulation and spatial misregistration robustness is only qualitatively described. Please provide more evidence or discussions.

    DCT 域调制与空间失配鲁棒性之间的关系仅进行了定性描述。请提供更多的证据或讨论。

6. In Fig. 3, the role of the frequency-domain difference feature should be further explained. It is unclear whether the difference operation may suppress useful modality-specific information.

    在图 3 中，应进一步解释频域差分特征的作用。尚不清楚差分操作是否会抑制有用的模态特异性信息。
