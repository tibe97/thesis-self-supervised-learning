


![GitHub](https://img.shields.io/github/license/lightly-ai/lightly)
![Unit Tests](https://github.com/lightly-ai/lightly/workflows/Unit%20Tests/badge.svg)
![codecov](https://codecov.io/gh/lightly-ai/lightly/branch/develop/graph/badge.svg?token=1NEAVROK3W)

The starting point of the code developed for my thesis work is the Lightly [repository](https://github.com/lightly-ai/lightly).
Lightly is a computer vision framework for self-supervised learning.

### Abstract
Unsupervised learning for computer vision has recently gained significant progress thanks to the self-supervised contrastive learning methods. Inspired from metric learning, the goal is learning good representations that are more robust and generalisable than the features learned in a supervised manner, when transferring on the downstream tasks. In this work we are going to see the most popular contrastive learning methods in self-supervised representa- tion learning, with a strong focus on the mining techniques devised for picking more informative positive and negative samples. We also propose a new ap- proach combining contrastive learning with an online clustering method in order to mine hard positive samples. We show that on fine-grained datasets, like the Stanford Dogs dataset with 120 dog breeds, it helps to learn more discriminative representations compared to other state-of-the-art contrastive methods.

### Features

Our method proposes a new way of sampling hard positives thanks to the usage of [SwAV](https://github.com/facebookresearch/swav) prototypes layer. Thanks to SwAV we can perform online clustering, which we leverage in order to obtain pseudo-labels for creating positive pairs by mining in the dataset.
Our framework features:

- positive mining module for mining positives which carry variation in viewpoint and deformation too
- combination of NNCLR and SwAV losses to guide the representation learning task
- supports custom backbone models for self-supervised pre-training


