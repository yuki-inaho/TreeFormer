# TreeFormer: Single-view Plant Skeleton Estimation via Tree-constrained Graph Generation 

Xinpeng Liu ${ }^{1}$ Hiroaki Santo ${ }^{1}$ Yosuke Toda ${ }^{2,3}$ Fumio Okura ${ }^{1}$<br>${ }^{1}$ Osaka University ${ }^{2}$ Phytometrics ${ }^{3}$ Nagoya University<br>\{liu.xinpeng, santo.hiroaki,okura\}@ist.osaka-u.ac.jp yosuke@phytometrics.jp

![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-01.jpg?height=437&width=1741&top_left_y=791&top_left_x=187)
Figure 1. We propose a method for single-image plant skeleton estimation combining learning-based graph generators with traditional graph algorithm (i.e., MST). The red lines show the predicted graph edges. Compared to (a) an unconstrained graph generator and (b) a naive tree-graph constraint implementation, (c) our method naturally imposes the constraint during the graph generation models' training. Our method can be directly applied to plant science and agricultural applications, such as (d) time-series reconstruction of botanical roots.


#### Abstract

Accurate estimation of plant skeletal structure (e.g., branching structure) from images is essential for smart agriculture and plant science. Unlike human skeletons with fixed topology, plant skeleton estimation presents a unique challenge, i.e., estimating arbitrary tree graphs from images. While recent graph generation methods successfully infer thin structures from images, it is challenging to constrain the output graph strictly to a tree structure. To this problem, we present TreeFormer, a plant skeleton estimator via tree-constrained graph generation. Our approach combines learning-based graph generation with traditional graph algorithms to impose the constraints during the training loop. Specifically, our method projects an unconstrained graph onto a minimum spanning tree (MST) during the training loop and incorporates this prior knowledge into the gradient descent optimization by suppressing unwanted feature values. Experiments show that our method accurately estimates target plant skeletal structures for multiple domains: Synthetic tree patterns, real botanical roots, and grapevine branches. Our implementations are available at https://github.com/huntorochi/ TreeFormer/.


## 1. Introduction

Skeletal structures of plants (e.g., branches and roots) are key information for analyzing plant traits in agriculture and plant science. In particular, single-view estimation of plant skeletons has potential benefits for various downstream tasks, such as high-throughput plant phenotyping [10, 18, 54] and plant organ segmentation [17, 47]. As a similar task, single-view estimation of human poses has been widely studied, e.g., OpenPose [11]. However, unlike human skeletons, which have a fixed graph topology, the plant skeleton is not organized because the number of joints and their relationships are unknown, posing a unique problem of estimating an arbitrary tree graph from an image.

The estimation of graph structure from images has been studied to extract thin structures such as road networks in satellite images [24, 61, 62]. Recent end-to-end models using recurrent neural networks (RNNs) [2], graph neural networks (GNNs) [36, 37, 43], or transformers [7, 29, 35, 40,55] show the ability to extract faithful unconstrained graph structures from images. However, inferring treeconstrained graphs with the existing graph generators becomes a non-trivial problem, where the output graph often violates the required constraints, as shown in Fig. 1(a). One reason for this difficulty is that tree graph generation, which
requires finding a set of graph edges that satisfy the constraint defined on the entire graph, naturally falls into combinatorial optimization. A simple way to impose the constraints on the graph generation is to convert the inferred unconstrained graphs to the closest graph that satisfies the given constraint using traditional graph algorithms such as Dijkstra's shortest path or minimum spanning tree (MST) algorithms. Such post-processing can work; however, because the graph generators are trained without any constraints, the output may be unrealistic, as shown in Fig. 1(b).

For tree graph generation from single images, we propose a simple yet effective way to integrate state-of-theart learning-based graph generation methods, which achieve high-quality image-based graph estimation, and traditional graph algorithms, which compute strictly constrained tree graphs. Specifically, we propose to project an unconstrained graph into a tree graph by a non-differentiable MST algorithm during each training loop. Our selective feature suppression (SFS) layer then converts the inferred unconstrained graph to the MST-based tree graph by a differentiable manner, thereby naturally incorporating the constraints into the graph generation.

By integrating our feature suppression layer with a state-of-the-art transformer-based graph generator, we develop TreeFormer, which infers tree structures from images capturing plants. We evaluate the effectiveness of TreeFormer on different classes of plant images: Synthetic tree patterns, real-world root, and grapevine branch images. The results show that our constraint-aware graph generator accurately estimates the target tree structures compared to baselines.

Contributions Our contributions are twofold: First, we propose a novel method that tightly integrates learningbased graph generation methods with traditional graph algorithms using the newly-proposed SFS layer, which modifies intermediate features in the network, effectively mimicking the behavior of the non-differentiable graph algorithms. Second, building upon our constrained graph generation method, we develop TreeFormer, the first end-to-end method inferring skeletal structures from a single plant image, which benefits the agriculture and plant science field.

## 2. Related Work

We propose constraining the graph structures given by image-based graph generators, whose primary goal is plant skeleton estimation. We, therefore, introduce the related work of plant skeleton estimation, graph generation from images, and constrained optimization for neural networks.

### 2.1. Plant skeleton estimation

Plant skeleton estimation is actively studied since it becomes a fundamental technique for downstream tasks related to plant phenotyping and cultivation [50].

3D plant skeleton estimation Several methods are proposed to derive plant skeletons from 3D observations [50]. These methods often use point clouds acquired by Li DAR [5, 60] or multi-view stereo (MVS) [54, 58]. Regardless of the 3D acquisition method, these works generally use a two-stage pipeline: Skeletonization [9] followed by graph optimization using MST or Dijkstra's algorithm [12, 15, 25, 47, 68], where the graph algorithms are required to convert a set of skeleton positions into a graph.

2D plant skeleton estimation Compared to 3D methods, skeleton estimation from a single 2D image poses significant technical challenges due to the lack of depth information and severe occlusions despite the simplicity of data acquisition. Like 3D methods, existing 2D methods use a two-stage process involving skeletonization and graph optimization. To extract the skeleton regions on 2D images, plant region segmentation is often used for plants with relatively thin leaves [19]. Similarly, a neural network that converts an input image into a map representing 2D skeleton positions is used to mitigate the occlusions [26]. To reason about the direction of intersecting branches, a recent work [18] proposes to use vector fields representing branch direction instead of mask images, similar to the Part Affinity Fields (PAFs) used in OpenPose [11].

Unlike existing two-stage methods, we propose an end-to-end method that directly infers a tree graph representing plant skeletons in a single image. Our experiments show that our end-to-end method achieves better accuracy than a recent two-stage method for 2D images.

### 2.2. Graph generation from images

Graph generation from images, sometimes called image-tograph generation, is studied for extracting thin structures (e.g., road networks) or relations (e.g., scene graphs) from images [2, 7, 14, 24, 29, 35-37, 40, 43, 55, 61, 62]. Recent learning-based methods often use object detectors, which detect graph nodes (e.g., intersections in road networks) from images, and then aggregate the combinations of node features to predict the edges defined between two nodes as binary (i.e., existence of edges) or categorical (e.g., classification of edge relations) values. Some studies use external knowledge [1, 13, 51, 53] to improve the results.

Graph generators have usually taken autoregressive methods (e.g., [7, 39, 44, 45, 66]) that output a graph by starting at an initial node and estimating neighboring graph nodes. Recent GNNs and transformers enable nonautoregressive graph generators [ $14,30,32,35,63,65$ ] simultaneously estimating the entire graph. Autoregressive methods are prone to errors during the estimation process, and the state-of-the-art non-autoregressive method, RelationFormer [55], performs better than autoregressive methods, especially for medium to large graphs.

A few recent studies consider graph generation with treegraph constraints in a different context. For the molecule structure estimation [4, 27, 28], these methods assume autoregressive graph generation, making it hard to work with complex and relatively large graphs like botanical plants.

### 2.3. Constrained optimization for neural networks

Constrained optimization is crucial for machine learning. In particular, introducing constraints in neural networks has become a recent trend [33].

Designing differentiable layers The most direct way to introduce additional constraints to neural networks is to make the constraints differentiable. In the continuous domain, it is known that a convex optimization can be implemented as a differentiable layer [3]. However, the design of differentiable layers for combinatorial optimization poses a significant challenge due to the difficulty of differentiation. Wilder et al. [57] propose a differentiable layer for linear programming (LP) problems using continuous relaxation. This method is extended to mixed integer linear programming (MILP) [16] by splitting the problem into multiple LPs. MST, which we want to use as constraints, is known to be transformed into the class of MILP [48, 52]. However, using differentiable layers for these complex combinatorial problems requires exponential computation time [33] to obtain the exact solution and is practically unrealistic.

Reparameterization for constrained optimization If the constraint function is difficult to differentiate, a simple alternative is to project unconstrained inferences or model parameters into constrained space, which can be considered a use of reparameterization [31].

In gradient descent optimization, methods projecting unconstrained optimization parameters (e.g., model parameters in neural networks) to the closest ones satisfying the given constraint are called projected gradient descent (PGD). PGD is often used for traditional optimization problems, directly optimizing the input variables [21,59]. While PGD can be used for neural network optimization, such as for generating adversarial examples [46], designing projection functions for neural networks is challenging. It requires mapping a large number of model parameters into a space satisfying complex constraints, where the constraints are often more naturally defined on the model output.

Instead of designing a projection function for the model parameters, the model's output can be projected onto the subspace that satisfies the constraints during the training loop. Since reparameterization for model output can be easily integrated with existing neural network models, they are often used for domain-specific applications such as coded aperture optimization with hardware constraint [64] and internal organ segmentation with given parametric shape models [8,38]. We take this approach in our SFS layer,
easily plugging it into off-the-shelf end-to-end graph generation methods without preparing the differentiable implementation of the constraints (i.e., MST algorithm).

## 3. Tree-constrained Graph Generation

We here describe the constrained graph generation method. Figure 2 summarizes the proposed SFS layer, which casts the original unconstrained edge probabilities to the constrained domain.

### 3.1. Problem statement

We here consider a simple setting of neural-network-based graph generation, where the model outputs the prediction of the edge probabilities (i.e., the edge exists or not) defined for a pair of nodes, while this can be extended to a multiclass classification setting straightforwardly.

Our goal is to design a tree-constrained graph generator $\mathcal{F}$ that converts a given image $I$ to a tree graph $G$ as

$$
\begin{equation*}
G=(V, E)=\mathcal{F}(I) \quad \text { s.t. } \quad E \in E_{\text {tree }}, \tag{1}
\end{equation*}
$$

where the graph $G$ consists of a set of nodes (or objects) $V$ and edges (or relations) $E$. Here, $E_{\text {tree }}$ denotes all possible edge patterns forming a tree graph given the set of nodes $V$.

We consider a (non-differentiable) projection function $\mathcal{P}$ that maps an unconstrained graph predicted by graph generators to the constrained graph $G$. Let the edge probabilities defined between each node pairs as $\left\{\hat{\mathbf{y}}_{(i, j)}\right\}_{(i, j) \in V \times V}$.

$$
\begin{equation*}
\hat{\mathbf{y}}_{(i, j)}=\left[\hat{y}_{(i, j)}^{+}, \hat{y}_{(i, j)}^{-}\right]^{\top} \quad \text { s.t. } \quad\left\|\hat{\mathbf{y}}_{(i, j)}\right\|_{1}=1, \tag{2}
\end{equation*}
$$

in which $\hat{y}_{(i, j)}^{+}$and $\hat{y}_{(i, j)}^{-}$respectively denote the edge existence and non-existence probabilities. The projection function $\mathcal{P}$ then read as

$$
\begin{equation*}
(V, E)=\mathcal{P}_{E \in E_{\text {tree }}}\left(V,\left\{\hat{\mathbf{y}}_{(i, j)}\right\}\right) \tag{3}
\end{equation*}
$$

which are given by traditional graph algorithms with combinatorial optimization. We assume the projection function $\mathcal{P}$ converts the existence probability (or category prediction) of graph edges while leaving the graph nodes $V$ unchanged. A typical example of $\mathcal{P}$ can be designed using the MST algorithm we use in our TreeFormer implementation, which projects an arbitrary graph into a tree structure by modifying the existence of graph edges based on the costs defined between each pair of nodes.

We aim to develop a differentiable function $\mathcal{R}$ that mimics the non-differentiable projection $\mathcal{P}$. Plugging with the unconstrained graph generator $\hat{\mathcal{F}}$, Eq. (1) is rewritten as

$$
\begin{align*}
& G=(V, E)=\mathcal{R}_{E \in E_{\text {tree }}}\left(V,\left\{\hat{\mathbf{y}}_{(i, j)}\right\}\right) \\
& \quad\left(V,\left\{\hat{\mathbf{y}}_{(i, j)}\right\}\right)=\hat{\mathcal{F}}(I) \tag{4}
\end{align*}
$$

where the whole process is differentiable.

![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-04.jpg?height=503&width=1722&top_left_y=254&top_left_x=200)
Figure 2. Overview of reparameterization layer that can be easily plugged into off-the-shelf graph generators. Given unconstrained edge predictions by graph generators, our method projects it to the closest constrained graph (i.e., tree) using a non-differentiable MST algorithm. Comparing constrained and unconstrained edges, unwanted edge features are selectively suppressed so that the graph becomes the tree.

### 3.2. SFS layer

Here, we describe an implementation of the SFS layer for constrained graph generation. As described in Eq. (2), the unconstrained graph generator $\hat{\mathcal{F}}$ computes the probability of unconstrained edge existence between the $i$-th and $j$-th nodes, $\hat{\mathbf{y}}_{(i, j)}=\left[\hat{y}_{(i, j)}^{+}, \hat{y}_{(i, j)}^{-}\right]^{\top}$. In neural networks, $\hat{\mathbf{y}}_{(i, j)}$ is usually computed through the softmax activation $\sigma$ applied to the output feature vector of the final layer $\hat{\mathbf{f}}_{(i, j)}=\left[\hat{f}_{(i, j)}^{+}, \hat{f}_{(i, j)}^{-}\right]^{\top} \in \mathbb{R}^{2}$ as

$$
\begin{equation*}
\hat{\mathbf{y}}_{(i, j)}=\sigma\left(\hat{\mathbf{f}}_{(i, j)}\right) . \tag{5}
\end{equation*}
$$

The set of unconstrained graph edges $\hat{E}$ are then obtained by comparing the edge existence probabilities as

$$
\begin{equation*}
\hat{E}=\left\{(i, j) \mid \hat{y}_{(i, j)}^{+}>\hat{y}_{(i, j)}^{-}\right\} \tag{6}
\end{equation*}
$$

in which $\hat{E}$ records node pairs where the edge exists.
Suppose the projection function $\mathcal{P}$ converts the set of unconstrained edge probabilities $\left\{\hat{\mathbf{y}}_{(i, j)}\right\}$ to a set of constrained edges $E$. Let the difference of two sets be $E^{+}= E-\hat{E}$ and $E^{-}=\hat{E}-E$, denoting the sets of edges newly added and removed by the projection. To mimic discrete (and non-differentiable) inferences by $\mathcal{P}$ in differentiable end-to-end learning, we modify the edge features corresponding to $E^{+} \cup E^{-}$in the differentiable forward process.

Specifically, what we want to get is the edge probabilities that approximate the constrained edges $E$, denoted as

$$
\mathbf{y}_{(i, j)}= \begin{cases}{[1-\epsilon, \epsilon]^{\top}} & \left((i, j) \in E^{+}\right)  \tag{7}\\ {[\epsilon, 1-\epsilon]^{\top}} & \left((i, j) \in E^{-}\right) \\ {\left[\hat{y}_{(i, j)}^{+}, \hat{y}_{(i, j)}^{-}\right]^{\top}} & (\text { otherwise })\end{cases}
$$

When $\epsilon$ is small enough, the constrained output $\mathbf{y}_{(i, j)}$ perfectly mimics the output by the projection function $\mathcal{P}$. However, the direct modification of the edge probabilities
naturally disconnects the computation graph. Therefore, we modify the unconstrained feature vector $\hat{\mathbf{f}}_{(i, j)}$ so that the corresponding edge probabilities $\mathbf{y}_{(i, j)}$ follows Eq. (7). Specifically, since $\mathbf{y}_{(i, j)}$ is computed through the softmax function $\sigma$, it is achieved via the following minimal modification that selectively suppresses the feature values by replacing them with a constant ${ }^{1}$ as

$$
\begin{array}{ll}
f_{(i, j)}^{-}:=-\Lambda & \left((i, j) \in E^{+}\right) \\
f_{(i, j)}^{+}:=-\Lambda & \left((i, j) \in E^{-}\right) \tag{8}
\end{array}
$$

where $\Lambda$ is assumed to be large enough to make $\exp (-\Lambda) \sim$ 0 . Given modified features $\mathbf{f}_{(i, j)}=\left[f_{(i, j)}^{+}, f_{(i, j)}^{-}\right]^{\top}$, the softmax activation $\sigma$ normalizes and converts them to edge probability $\mathbf{y}_{(i, j)}$.

In summary, from Eqs. (5) and (S5), the constrained edge prediction between $i$-th and $j$-th nodes, $\mathbf{y}_{i j}=\left[y_{i j+}, y_{i j-}\right]^{\top}$, is obtained as

$$
\mathbf{y}_{(i, j)}= \begin{cases}\sigma\left(\left[\hat{f}_{(i, j)}^{+},-\Lambda\right]^{\top}\right) & \left((i, j) \in E^{+}\right)  \tag{9}\\ \sigma\left(\left[-\Lambda, \hat{f}_{(i, j)}^{-}\right]^{\top}\right) & \left((i, j) \in E^{-}\right) \\ \sigma\left(\left[\hat{f}_{(i, j)}^{+}, \hat{f}_{(i, j)}^{-}\right]^{\top}\right) & (\text { otherwise })\end{cases}
$$

After the reparameterization, the set of edges computed from $\left\{\mathbf{y}_{(i, j)}\right\}$ in the same way as Eq. (S2) is guaranteed to be equal to $E$ inferred by the discrete projection function $\mathcal{P}$ when $\Lambda$ is large enough.

### 3.3. Analysis

The common auto differentiation libraries automatically compute the gradient of the SFS layer. Although it can disconnect the computation path at a feature, since we keep at least one of the original features (either $\hat{f}_{(i, j)}^{+}$or $\hat{f}_{(i, j)}^{-}$), the backpropagation path to the backbone graph generation network is not disconnected ${ }^{2}$. Here, we briefly analyze the be-

[^0]havior of the SFS layer. The supplementary materials provide a detailed analysis, including mathematical proofs.

When using the cross-entropy loss $\mathcal{L}_{\mathrm{CE}}$ to evaluate the availability of the graph edges, the derivative to be backpropagated to the backbone graph generator is approximated as ${ }^{3}$

$$
\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial \hat{\mathbf{f}}} \sim\left\{\begin{array}{cl}
{\left[1-t^{+}, \quad 0 \quad\right]^{\top}} & \left((i, j) \in E^{+}\right)  \tag{10}\\
{\left[\begin{array}{c}
0,1-t^{-} \\
]^{\top}
\end{array}\right]} & \left((i, j) \in E^{-}\right) \\
{\left[y^{+}-t^{+}, y^{-}-t^{-}\right]^{\top}} & (\text { otherwise })
\end{array}\right.
$$

where $\mathbf{t}=\left[t^{+}, t^{-}\right]^{\top} \in\{0,1\}^{2}$ denotes the ground truth edge existence and non-existence for the node pair $(i, j)$. Our method modifies the computation graph of the network when the MST algorithm disagrees with the output of graph generation model (i.e., $(i, j) \in E^{+} \cup E^{-}$), but in different ways for derivatives of each feature value $\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial f^{+}}$or $\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial f^{-}}$.

Without loss of generality, we consider the case when the MST algorithm adds an edge, i.e., $(i, j) \in E^{+}$. When the MST correctly modify the edge availability (i.e., $t^{+}=1$ ), the gradient vector becomes small, $\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial \hat{\mathbf{f}}} \sim \mathbf{0}$, which is the behavior we expect. On the other hand, if the MST incorrectly adds the edge (i.e., $t^{+}=0$ ), the gradient becomes $[1,0]^{\top}$, which strongly penalizes the positive edge probability, where the norm of the gradient vector is always larger than unconstrained ones ${ }^{4}$. Therefore, the behavior of our simple reparameterization strategy is reasonable in practice.

## 4. TreeFormer: A Plant Skeleton Estimator

We develop TreeFormer, an implementation of the SFS layer to a state-of-the-art graph generator. This section first recaps the graph generator [55] and then details how we introduce tree structure constraint.

### 4.1. RelationFormer: A brief recap

RelationFormer [55] is the state-of-the-art nonautoregressive graph generation method. This method uses an end-to-end architecture that combines an object (node) detector and relation (edge) predictor, which shows superior performance for unconstrained graph generation. The object detection part is based on deformable DETR [67], which is trained to extract graph nodes (e.g., objects) and global features from a given image. Specifically, given the extracted image features, the transformer decoder outputs a fixed number of object queries ([obj]-tokens) representing each of the nodes and a relation query ([rtn]-token) describing the global features, including node relations.

The relation prediction head outputs the relationship (i.e., edge existence or category) from the detected pairs of objects (i.e., [obj]-tokens) and the global relation (i.e.,

[^1][rtn]-tokens). This module is implemented as a multi-layer perceptron (MLP) headed by layer normalization [6]. RelationFormer is trained using the sum of loss functions related to object detection and edge (relation) estimation, where edge (relation) loss $\mathcal{L}_{\text {edge }}{ }^{5}$ evaluates the edge existence or category between node pairs using cross-entropy loss.

### 4.2. Tree-constrained graph generation

To introduce the tree structure constraint, we use Kruskal's MST algorithm [34] implemented in NetworkX ${ }^{6}$. To extract a tree from an unconstrained graph predicted by RelationFormer, we use the edge non-existence probabilities $\left\{\hat{y}_{(i, j)}^{-}\right\}$ as the edge cost for the MST algorithm to span the tree on edges with higher existence probabilities.

We implement the SFS layer on top of the relation prediction head in the RelationFormer. Specifically, the output features from the MLP after layer normalization are regarded as unconstrained features $\{\hat{\mathbf{f}}\}$. In our experiments, we use $\Lambda=10$ during training, where $\exp (-\Lambda)=4.5 \times 10^{-5}$. We show an ablation study changing $\Lambda$ in the supplementary materials.

Loss function Our SFS layer affects the evaluation of the edge loss $L_{\text {edge }}$ in the graph generator, while the computation of other loss functions, such as for node detection, remains the same as in the original implementation. Our implementation uses both loss functions for original (unconstrained) and constrained edges. Denoting the groundtruth edges as $E_{\mathrm{GT}}=\left\{(i, j) \mid t_{(i, j)}^{+}>t_{(i, j)}^{-}\right\}$, where $\mathbf{t}_{(i, j)}=\left[t_{(i, j)}^{+}, t_{(i, j)}^{-}\right]^{\top} \in\{0,1\}^{2}$, the loss function for edge availability $\mathcal{L}_{\text {edge }}$ is modified as follows

$$
\begin{equation*}
\mathcal{L}_{\text {edge }}=\underbrace{\sum_{(i, j)} \mathcal{L}_{\mathrm{CE}}\left(\hat{\mathbf{y}}_{(i, j)}, \mathbf{t}_{(i, j)}\right)}_{\mathcal{L}_{\text {unconst }}}+\underbrace{\sum_{(i, j)} \mathcal{L}_{\mathrm{CE}}\left(\mathbf{y}_{(i, j)}, \mathbf{t}_{(i, j)}\right)}_{\mathcal{L}_{\text {const }}}, \tag{11}
\end{equation*}
$$

where $\mathcal{L}_{\mathrm{CE}}$ denotes the cross-entropy loss.

## 5. Experiments

To assess the effectiveness of the proposed method and TreeFormer implementation, we perform experiments using synthetic and real image datasets.

### 5.1. Datasets

We use one synthetic and two real datasets, where examples are shown in Fig. 3. Supplementary materials describe the details of the datasets.

Synthetic dataset To systematically demonstrate the performance of our method, we perform an experiment using a large synthetic dataset. We automatically generate images

[^2]![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-06.jpg?height=335&width=835&top_left_y=243&top_left_x=195)
Figure 3. Example images from the dataset we used for our experiments. Annotated graphs are superimposed. Yellow dots and red lines indicate nodes and edges.

of tree patterns using pre-defined rules of Lindenmayer systems (L-system) [20, 42], which generate structural patterns using recursive processes. We add randomness of branching patterns, branch length, and joint angles to increase the dataset variation. The number of nodes in the graph is controlled at less than 100 . The resolution of the generated images is $512 \times 512$ pixels. We generated 100000 images for training, 20000 for validation, and 20000 for testing.

Root dataset We use photographs of early-growing roots of Arabidopsis, which are often important targets of analysis in plant science. In this dataset, the graph structures are manually annotated. The dataset contains 781 root images, and we randomly divide them into 625 training, 78 validation, and 78 test images. Each graph contains up to 117 nodes. The image resolution is $570 \times 190$ pixels. We use data augmentation involving rotation, flipping, and cropping for the training dataset, which collectively expands the training dataset to 62,500 images.

Grapevine dataset [18] We use 3D2cut Single Guyot Dataset [18] containing grapevine tree images captured in an agricultural field with annotated branch patterns. The dataset contains relatively complex structures; the graph contains up to 205 nodes. The resolution is $504 \times 378$ pixels. The dataset contains 1503 images, and we use the dataset split the same as [18], where 1185 images are for training, and 63 and 255 images are for validation and testing, respectively. We use data augmentation in the same manner as the root dataset, resulting in 118, 500 training images.

### 5.2. Evaluation metrics

We use different metrics to capture spatial similarity alongside the topological similarity of the predicted graphs.

Street mover's distance (SMD) [7] SMD is a metric to assess the accuracy of the positions of graph edges, which is computed as the Wasserstein distance between the predicted and the ground truth edges. In our implementation, the distance is computed between densely sampled points on the edges, which is the same procedure as in the original paper proposing the SMD [7].

TOPO score [23] We compute the TOPO scores to evaluate the topological mismatch of the output graph. This metric consists of the precision, recall, and F1 scores of the graph nodes, which are evaluated considering the edge topology. We use the implementation used in Sat2Graph paper [24], while we only evaluate the nodes with the degree $\neq 2$ that affect the tree structure, i.e., we only evaluate joint and leaf nodes in the graphs.

Tree rate To evaluate how well the output graph satisfies the constraint, we calculate the probability that the output graph forms a tree structure. While it is obvious that the tree rate becomes $100 \%$ for constrained methods, including ours, we are interested in how well the output of the unconstrained graph generation model can reflect the constraint by training on datasets that contain only tree graphs.

### 5.3. Baselines

Since the constrained graph generation task is new in this paper, there are few established baseline methods. We compare our method with the state-of-the-art methods for 2D plant structure estimation and unconstrained graph generation. Also, as an ablation study, we compare a simpler alternative to our method. Supplementary materials provide additional comparisons with other baseline methods, including autoregressive graph generation.

Two-stage [18] We implement a 2D plant skeleton estimation method based on a two-stage method involving skeletonization and graph optimization with reference to [18]. Specifically, vector fields of branch directions are generated by a neural network, followed by graph optimization to generate branch structure, in which we find our implementation outperforms the naive re-implementation of the existing method [18]. Specific implementations and analyses are described in the supplementary materials.

Unconstrained [55] We compare the state-of-the-art (unconstrained) graph generation method, RelationFormer [55]. This method is identical to our method without applying the tree structure constraint.

Test-time constraint As a straightforward implementation of constrained graph generation, we apply MST only in the inference phase, where the graph generator is trained using the same procedure as the unconstrained method.

### 5.4. Implementation details

For RelationFormer in our method and the baseline comparison, we use the official implementation ${ }^{7}$ on PyTorch. For other hyperparameters, we follow the original RelationFormer implementation used for road network extraction.

[^3]| Input image | Ground truth | Two-stage | Unconstrained | Test-time constraint | Ours |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-07.jpg?height=227&width=202&top_left_y=314&top_left_x=226) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-07.jpg?height=227&width=201&top_left_y=314&top_left_x=516) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-07.jpg?height=229&width=203&top_left_y=312&top_left_x=798) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-07.jpg?height=233&width=213&top_left_y=308&top_left_x=1076) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-07.jpg?height=233&width=207&top_left_y=308&top_left_x=1370) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-07.jpg?height=227&width=201&top_left_y=314&top_left_x=1663) |
| ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-07.jpg?height=250&width=192&top_left_y=572&top_left_x=221) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-07.jpg?height=250&width=200&top_left_y=572&top_left_x=501) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-07.jpg?height=250&width=191&top_left_y=572&top_left_x=794) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-07.jpg?height=250&width=205&top_left_y=572&top_left_x=1069) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-07.jpg?height=250&width=196&top_left_y=572&top_left_x=1366) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-07.jpg?height=250&width=202&top_left_y=572&top_left_x=1644) |

Figure 4. Visual results for the synthetic tree pattern dataset. From left to right: Input images, results of the two-stage method (similar to [18]), the unconstrained method (identical to RelationFormer [55]), a naive implementation with test-time constraint, and ours are shown. We translucently overlay the estimated and ground truth edges with red and blue lines, respectively. While all methods accurately detect nodes, only our method accurately predicts the availability of edges from given images compared to the baseline methods.

Table 1. Quantitative results. Our method significantly improves both the shape and topology of the predicted graph while enforcing the given constraints. The best scores are highlighted bold.
| Dataset | Method | SMD $\downarrow$ | TOPO score $\uparrow$ |  |  | Tree rate [\%] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  |  |  | Prec. | Rec. | F1 |  |
| Synthetic | Two-stage [18] | $1.91 \times 10^{-3}$ | 0.940 | 0.886 | 0.912 | 100.0 |
|  | Unconstrained [55] | $1.43 \times 10^{-5}$ | 0.978 | 0.929 | 0.953 | 36.2 |
|  | Test-time constraint | $6.26 \times 10^{-6}$ | 0.977 | 0.953 | 0.965 | 100.0 |
|  | Ours | $\mathbf{4 . 7 8} \boldsymbol{\times} \mathbf{1 0}^{\boldsymbol{-} \mathbf{6}}$ | 0.986 | 0.968 | 0.977 | 100.0 |
| Root | Two-stage [18] | $4.83 \times 10^{-4}$ | 0.767 | 0.732 | 0.749 | 100.0 |
|  | Unconstrained [55] | $1.19 \times 10^{-4}$ | 0.831 | 0.633 | 0.719 | 35.9 |
|  | Test-time constraint | $1.52 \times 10^{-4}$ | 0.829 | 0.771 | 0.799 | 100.0 |
|  | Ours | $8.82 \times 10^{-5}$ | 0.861 | 0.807 | 0.833 | 100.0 |
| Grapevine | Two-stage [18] | $4.24 \times 10^{-4}$ | 0.677 | 0.589 | 0.630 | 100.0 |
|  | Unconstrained [55] | $1.45 \times 10^{-4}$ | 0.963 | 0.559 | 0.708 | 0.0 |
|  | Test-time constraint | $1.47 \times 10^{-4}$ | 0.896 | 0.840 | 0.867 | 100.0 |
|  | Ours | $\mathbf{1 . 0 3} \times \mathbf{1 0}^{-\mathbf{4}}$ | 0.899 | 0.843 | 0.870 | 100.0 |


We used early stopping for all datasets and methods by selecting the model with the best validation performance and terminating training after 30 epochs without improvement. The training of our method takes approximately 141 hours for the synthetic dataset, 10 hours for the root dataset, and 98 hours for the grapevine dataset, all conducted on eight NVIDIA RTX A100 GPUs.

### 5.5. Results on synthetic dataset

Figure 4 shows visual results for the synthetic dataset, where the red and blue lines indicate the predicted and ground truth edges, respectively. Since they are shown translucently, if the estimated edge overlaps the true edge, it is displayed in purple. Similarly, cyan and yellow dots indicate the nodes, which merge into green if correctly estimated. From the results, all methods correctly estimate the node positions. The existing unconstrained method outputs isolated edges and cycles. Although the two-stage and test-time constraint methods enforce the tree structure con-
straint, they often produce incorrect edges. Compared to the baselines, our method accurately generates the graph edges.

The above trend can be quantitatively confirmed in Table 1. The unconstrained method produces tree structures with only about $30 \%$ probability, even though all the training graphs form tree structures. Although introducing the test-time constraint and two-stage methods improves the shape and topology, there are still many incorrect estimates. Compared to those baseline methods, our method significantly improves both edge positions and graph topology.

### 5.6. Results on real datasets

Figure 5 show results of skeleton estimation for two realworld datasets. For these figures, red lines, yellow dots, and cyan dots indicate the edges, nodes, and keypoints (i.e., joints and leaf nodes), respectively. In agreement with the synthetic results, our method predicts visually better structures, while the unconstrained model hardly produces tree structures. The method with test-time constraint clearly produces false edges, as shown in the results for the grapevine images. The two-stage method is often sensitive to the node detection error, leading to unnecessary (cf. Fig. 5a) or missing (cf. Fig. 5b) keypoints. In these practical settings, our end-to-end pipeline especially benefits from the simultaneous optimization of edge and node detection, resulting in faithful predictions at both nodes and edges for real-world datasets.

The quantitative results in Table 1 confirm the advantage of our method for real-world scenes. Our method shows particularly compelling results on grapevine datasets with relatively complex branching structures, outperforming the second-best method (test-time constraint) by approximately 30 \% improvement on the edge accuracy evaluated by SMD.

![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-08.jpg?height=1058&width=1741&top_left_y=241&top_left_x=203)
Figure 5. Visual results for the real image datasets. Red lines, yellow dots, and cyan dots indicate the predicted graph edges, nodes, and keypoints (i.e., joints and leaf nodes). Our method accurately estimates the target plant structures compared with baseline methods, demonstrating the applicability of our method for practical uses in plant science and agriculture.

![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-08.jpg?height=533&width=828&top_left_y=1460&top_left_x=200)
Figure 6. Results for the additional test images of (a) a grapevine tree under a natural background and (b-d) other tree species.

Generalization ability We test our model on additional test datasets to validate the out-of-domain performance of our method, using the model trained on the Grapevine dataset. Although the model is trained with grapevine trees with few background textures, it successfully works for grapevine images with background textures (Fig. 6(a)) and for other tree species (Fig. 6(b-d)). These results highlight the generalizability of our method.

## 6. Conclusion

We present the first attempt at tree-constrained graph generation from a single image, especially for plant skeleton estimation. We combine modern learning-based graph generators and traditional graph algorithms via the SFS layer, easily integrated with off-the-shelf graph generators.
Limitations We use graph algorithms during each training iteration, taking a longer training time than unconstrained methods, where fast GPU-based MST implementations (e.g., [56]) can improve computational performance. The success of our method depends on the accuracy of the underlying graph generation model, as we see a few undetected nodes in the visual results. Unlike universal human skeleton estimation such as OpenPose [11], our method requires domain-specific training due to the excessive variety of real-world plant appearances and structures, although we show certain generalizability in our experiments.
Acknowledgements We thank Professor Yasuyuki Matsushita for insightful discussions throughout the study. We also thank Momoko Takagi, Manami Okazaki, and Professor Kei Hiruma for providing us with root images. This work was partly supported by JSPS KAKENHI Grant Numbers JP22K17910, JP23H05491, and JP21H03466, and JST FOREST Grant Number JPMJFR206F.

# TreeFormer: Single-view Plant Skeleton Estimation via Tree-constrained Graph Generation 

Supplementary Material

This supplementary material provides additional information, including details of our SFS layer (Sec. A), dataset details (Sec. B), implementation details of the baseline methods (Sec. C), performance analysis of our method (Sec. D), other design choices (Sec. E), and more visual results (Sec. F).

## A. Details of SFS layer

## A.1. Motivation

Our method infers a tree graph via MST as formulated in Eq. (1)-Eq. (4) in the main paper. Our task's goal is to optimize the graph generation network so that the final output (i.e., tree graph via MST) becomes similar to the ground-truth tree graph. Since MST modifies the edge availability in unconstrained inferences, the unconstrained methods evaluating the unconstrained graph edges are indirect. Instead, our method directly evaluates the quality of the final output tree by mimicking MST. While experiments highlight our method's benefit, the following theoretical analysis also supports this intuition.

## A.2. Derivation

This section details the derivation of Eq. (10) in the main paper. To make this material self-contained, we repeat several descriptions in the main paper.

As discussed in the main paper, we consider the edge probabilities $\hat{\mathbf{y}}_{(i, j)}=\left[\hat{y}_{(i, j)}^{+}, \hat{y}_{(i, j)}^{-}\right]^{\top}$ is usually computed through the softmax activation $\sigma$ applied to the output feature vector of the final layer $\hat{\mathbf{f}}_{(i, j)}=\left[\hat{f}_{(i, j)}^{+}, \hat{f}_{(i, j)}^{-}\right]^{\top}$ as

$$
\begin{align*}
& \hat{\mathbf{y}}_{(i, j)}=\sigma\left(\hat{\mathbf{f}}_{(i, j)}\right) \\
& =\left[\frac{\exp \left(\hat{f}_{(i, j)}^{+}\right)}{\exp \left(\hat{f}_{(i, j)}^{+}\right)+\exp \left(\hat{f}_{(i, j)}^{-}\right)}, \frac{\exp \left(\hat{f}_{(i, j)}^{-}\right)}{\exp \left(\hat{f}_{(i, j)}^{+}\right)+\exp \left(\hat{f}_{(i, j)}^{-}\right)}\right]^{\top} . \tag{S1}
\end{align*}
$$

The set of unconstrained graph edges $\hat{E}$ are then obtained by comparing the edge existence probabilities as

$$
\begin{equation*}
\hat{E}=\left\{(i, j) \mid \hat{y}_{(i, j)}^{+}>\hat{y}_{(i, j)}^{-}\right\}, \tag{S2}
\end{equation*}
$$

in which $\hat{E}$ records node pairs where the edge exists.
Suppose the projection function $\mathcal{P}$ converts the set of unconstrained edge probabilities $\left\{\hat{\mathbf{y}}_{(i, j)}\right\}$ to a set of constrained edges $E$. Let the difference of two sets be $E^{+}= E-\hat{E}$ and $E^{-}=\hat{E}-E$, denoting the sets of edges newly
added and removed by the projection. To mimic the discrete (and non-differentiable) inferences by $\mathcal{P}$ in the differentiable end-to-end learning, we modify the edge features corresponding to $E^{+} \cup E^{-}$in the differentiable forward process. Here, we want to get the edge probabilities that approximate the constrained edges $E$, which can be denoted as

$$
\begin{align*}
\mathbf{y}_{(i, j)} & = \begin{cases}{\left[\begin{array}{cc}
1 & 0 \\
0 & 1
\end{array}\right]^{\top}} & \left((i, j) \in E^{+}\right) \\
{\left[\begin{array}{c}
0 \\
{\left[\hat{y}_{(i, j)}^{+}, \hat{y}_{(i, j)}^{-}\right.}
\end{array}\right]^{\top}} & \left((i, j) \in E^{-}\right)\end{cases}  \tag{S3}\\
& \sim\left\{\begin{array}{cc}
{[1-\epsilon, \epsilon]^{\top}} & \left((i, j) \in E^{+}\right) \\
{[\epsilon, 1-\epsilon]^{\top}} & \left((i, j) \in E^{-}\right) \\
{\left[\hat{y}_{(i, j)}^{+}, \hat{y}_{(i, j)}^{-}\right]^{\top}} & (\text { otherwise }) .
\end{array}\right. \tag{S4}
\end{align*}
$$

When $\epsilon$ is small enough, the constrained output $\mathbf{y}_{(i, j)}$ perfectly mimics the output by the projection function $\mathcal{P}$. Our goal is to modify the feature vector $\hat{\mathbf{f}}_{(i, j)}$ so that it makes the probabilities as Eq. (S4) through the softmax activation.

In the SFS layer, we replace the features as

$$
\begin{array}{ll}
f_{(i, j)}^{-}:=-\Lambda & \left((i, j) \in E^{+}\right)  \tag{S5}\\
f_{(i, j)}^{+}:=-\Lambda & \left((i, j) \in E^{-}\right),
\end{array}
$$

where $\Lambda$ is assumed to be large enough. Given modified features $\mathbf{f}_{(i, j)}=\left[f_{(i, j)}^{+}, f_{(i, j)}^{-}\right]^{\top}$, the softmax activation $\sigma$ normalizes and converts them to edge probability $\mathbf{y}_{(i, j)}$ as

$$
\mathbf{y}_{(i, j)}= \begin{cases}\sigma\left(\left[\hat{f}_{(i, j)}^{+},-\Lambda\right]^{\top}\right) & \left((i, j) \in E^{+}\right)  \tag{S6}\\ \sigma\left(\left[-\Lambda, \hat{f}_{(i, j)}^{-}\right]^{\top}\right) & \left((i, j) \in E^{-}\right) \\ \sigma\left(\left[\hat{f}_{(i, j)}^{+}, \hat{f}_{(i, j)}^{-}\right]^{\top}\right) & (\text { otherwise }) .\end{cases}
$$

Without loss of generality, we discuss the case in $(i, j) \in E^{+}$. Substituting Eq. (S6) into Eq. (S1) yields

$$
\begin{aligned}
\mathbf{y}_{(i, j)} & =\left[\frac{\exp \left(\hat{f}_{(i, j)}^{+}\right)}{\exp \left(\hat{f}_{(i, j)}^{+}\right)+\exp (-\Lambda)}, \frac{\exp (-\Lambda)}{\exp \left(\hat{f}_{(i, j)}^{+}\right)+\exp (-\Lambda)}\right]^{\top} \\
& =\left[\frac{\exp \left(\hat{f}_{(i, j)}^{+}\right)}{\exp \left(\hat{f}_{(i, j)}^{+}\right)+\epsilon^{\prime}}, \frac{\epsilon^{\prime}}{\exp \left(\hat{f}_{(i, j)}^{+}\right)+\epsilon^{\prime}}\right]^{\top} \quad\left((i, j) \in E^{+}\right),
\end{aligned}
$$

where $\epsilon^{\prime}=\exp (-\Lambda) \sim 0$ when $\Lambda$ is large enough, leading to $\mathbf{y}_{(i, j)}=[1-\epsilon, \epsilon]^{\top}$ as in Eq. (S4) by denoting $\epsilon=\frac{\epsilon^{\prime}}{\exp \left(\hat{f}_{(i, j)}^{+}\right)+\epsilon^{\prime}} \sim 0$.

Table S1. Case-by-case analyses of our reparameterization layer. For the columns of unconstrained features $\hat{\mathbf{f}}$, constrained prediction $\mathbf{y}$, and the ground truth edge availability $\mathbf{t}$, the table shows the index of the larger element. For example, the column $\hat{\mathbf{f}}$ will be + when the edge feature for the positive edge availability is larger, i.e., $\hat{f^{+}}>\hat{f^{-}}$. The column ( $i, j$ ) displays $E^{+}$or $E^{-}$if the MST algorithm modifies the edge availability (in which the rows are also highlighted). For the remaining columns, $\uparrow$ and $\downarrow$ denote each value becoming (relatively) large or small, respectively.
| Case | Feats \& probs |  |  | GT t | Loss $\mathcal{L}_{\text {CE }}$ | Approx. derivatives |  |  |  | Descriptions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  | $\hat{\mathrm{f}}$ | ( $i, j$ ) | y |  |  | $\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial \hat{f}^{+}}$ | $\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial \tilde{f}^{-}}$ | $\left\|\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial f^{+}}\right\|$ | $\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial f^{-}}$ |  |
| 1 | + |  | + | $[1,0]^{\top}$ | $\downarrow$ | $y^{+}-1$ | $y^{-}$ | $\downarrow$ | $\downarrow$ | Unmodified |
| 2 | + |  | + | $[0,1]^{\top}$ | $\uparrow$ | $y^{+}$ | $y^{-}-1$ | $\uparrow$ | $\uparrow$ | Unmodified |
| 3 | + | $E^{-}$ | - | $[1,0]^{\top}$ | $\uparrow$ | 0 | 1 | $\downarrow$ | $\uparrow$ | MST incorrectly modified |
| 4 | + | $E^{-}$ | - | $[0,1]^{\top}$ | $\downarrow$ | 0 | 0 | $\downarrow$ | $\downarrow$ | MST correctly modified |
| 5 | - |  | - | $[1,0]^{\top}$ | $\uparrow$ | $y^{+}-1$ | $y^{-}$ | $\uparrow$ | $\uparrow$ | Unmodified |
| 6 | - |  | - | $[0,1]^{\top}$ | $\downarrow$ | $y^{+}$ | $y^{-}-1$ | $\downarrow$ | $\downarrow$ | Unmodified |
| 7 | - | $E^{+}$ | + | $[1,0]^{\top}$ | $\downarrow$ | 0 | 0 | $\downarrow$ | $\downarrow$ | MST correctly modified |
| 8 | - | $E^{+}$ | + | $[0,1]^{\top}$ | $\uparrow$ | 1 | 0 | $\uparrow$ | $\downarrow$ | MST incorrectly modified |


## A.3. Detailed analysis

We describe a detailed analysis of our reparameterization layer. As described in (S6), the unconstrained edge feature between $i$ and $j$-th nodes $\hat{\mathbf{f}}_{(i, j)}=\left[\hat{f}_{(i, j)}^{+}, \hat{f}_{(i, j)}^{-}\right]^{\top}$ is converted to constrained prediction of the edge availability $\mathbf{y}_{(i, j)}=\left[y_{(i, j)}^{+}, y_{(i, j)}^{-}\right]^{\top}$ by selectively suppressing unwanted feature values.

When using the cross-entropy loss $\mathcal{L}_{\mathrm{CE}}$ to evaluate the availability of the graph edges, the derivative to be backpropagated to the backbone graph generator is ${ }^{8}$

$$
\begin{align*}
\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial \hat{\mathbf{f}}} & = \begin{cases}{\left[(1-\epsilon)-t^{+}, \quad 0 \quad\right]^{\top}} & \left((i, j) \in E^{+}\right) \\
{\left[\begin{array}{c}
0 \quad,(1-\epsilon)-t^{-} \\
{\left[y^{+}-t^{+}, \quad y^{-}-t^{-}\right.}
\end{array}\right]^{\top}} & \left((i, j) \in E^{-}\right)\end{cases}  \tag{S7}\\
& \sim\left\{\begin{array}{cc}
{\left[\begin{array}{c}
1-t^{+}, \quad 0 \\
0 \quad, \quad 1-t^{-}
\end{array}\right]^{\top}} & \left((i, j) \in E^{+}\right) \\
{\left[\begin{array}{cc}
0 & \left((i, j) \in E^{-}\right) \\
{\left[y^{+}-t^{+}, y^{-}-t^{-}\right]^{\top}} & (\text { otherwise }),
\end{array}\right.}
\end{array}\right. \tag{S8}
\end{align*}
$$

where $\mathbf{t}=\left[t^{+}, t^{-}\right]^{\top}$ denotes the ground truth edge existence and non-existence for the node pair $(i, j)$. Our method modifies the computation graph of the network when the MST algorithm does not agree with the output of the graph generation model (i.e., $(i, j) \in E^{+} \cup E^{-}$), but in different ways for derivatives of each feature value $\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial f^{+}}$or $\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial f^{-}}$.

Table S1 summarizes the case-by-case behavior, in which we can categorize the behaviors of the SFS layer into eight cases. Hereafter, we use $E^{*} \triangleq E^{+} \cup E^{-}$.
Case $(i, j) \notin E^{*}$ (Cases 1, 2, 5, 6) When the graph algorithm (i.e., MST) does not modify the edge availability (i.e., cases $1,2,5$, and 6 in the table), the behavior is the same as the usual cross-entropy loss for unconstrained edges.

[^4]Case $(i, j) \in E^{*} \quad \& \quad \mathbf{y} \sim \mathbf{t}$ (Cases 4, 7) In these cases, the MST algorithm correctly suppresses the unwanted features, where the constrained prediction $\mathbf{y}$ becomes the approximation of the ground-truth edge availability $\mathbf{t}$. The loss value becomes small, and the derivative is $\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial \hat{\mathrm{f}}} \sim \mathbf{0}$. This is natural since our constrained graph generator produces correct predictions.

Case $(i, j) \in E^{*} \& \mathbf{y} \nsim \mathbf{t}$ (Cases 3, 8) In these cases, MST incorrectly modifies the edge availability, i.e., the node pair ( $i, j$ ) belongs to $E^{+}$or $E^{-}$, but the constrained prediction $\mathbf{y}$ does not fit the ground truth $\mathbf{t}$. Here, we mathematically discuss the behavior in these cases. Without loss of generality, we focus on Case 3, where MST incorrectly removes an edge and compares the methods with and without tree-graph constraints using the SFS layer.

Case 3 (MST incorrectly removes an edge) The following discussions can be straightforwardly extended to Case 8, where MST incorrectly adds an edge.

## Conditions:

- Onnconstrained prediction (edge exists): $\hat{f}^{+}>\hat{f}^{-}$,
- MST removes the edge: $(i, j) \in E^{-}$,
- GT edge availability (edge exists): $\left[t^{+}, t^{-}\right]=[1,0]$.

Unconstrained method (without SFS layer) The gradient at $\hat{f}^{+}$by the unconstrained method is

$$
\begin{equation*}
\frac{\partial \mathcal{L}_{\text {unconst }}}{\partial \hat{f}^{+}}=\frac{\exp \left(\hat{f}^{+}\right)}{\exp \left(\hat{f}^{+}\right)+\exp \left(\hat{f}^{-}\right)}-1 . \tag{S9}
\end{equation*}
$$

Since $\hat{f}^{+}>\hat{f}^{-}$, it takes the value in the range of $(-0.5,0)$. Similarly,

$$
\begin{equation*}
\frac{\partial \mathcal{L}_{\text {unconst }}}{\partial \hat{f}^{-}}=\frac{\exp \left(f^{-}\right)}{\exp \left(f^{-}\right)+\exp \left(f^{+}\right)}-0 \tag{S10}
\end{equation*}
$$

thus the gradient at $\hat{f}^{-}$is inside ( $0,0.5$ ). Thus, the gradient vector $\frac{\partial \mathcal{L}_{\text {unconst }}}{\partial \hat{\mathbf{f}}}$ is always shorter than $[-0.5,0.5]^{\top}$ (corresponding to the special case $\hat{f}^{+}=\hat{f}^{-}$).

Constrained method (Ours) As described in Eqs. (S7) and (S8), our method yields the gradient as

$$
\begin{aligned}
& \quad \frac{\partial \mathcal{L}_{\text {const }}}{\partial \hat{f}^{+}}=0, \quad \frac{\partial \mathcal{L}_{\text {const }}}{\partial \hat{f}^{-}}=1-\epsilon \sim 1, \\
& \text { i.e., } \frac{\partial \mathcal{L}_{\text {const }}}{\partial \hat{\mathbf{f}}} \sim[0,1]^{\top} .
\end{aligned}
$$

Comparisons While both methods control the features to increase the edge availability, the relation of gradient vectors $\left\|\frac{\partial \mathcal{L}_{\text {const }}}{\partial \hat{\mathbf{f}}}\right\|>\left\|\frac{\partial \mathcal{L}_{\text {unconst }}}{\partial \hat{\mathbf{f}}}\right\|$ always holds, which means our

![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-11.jpg?height=487&width=833&top_left_y=243&top_left_x=195)
Figure S1. Atomic structures used for synthetic dataset generation. Three pre-defined initial structures are highlighted in purple. Eight pre-defined rewriting rules are used during the generation.

method strongly penalizes the incorrect estimates by MST by directly comparing the final estimation (i.e., tree graph) with the ground-truth edge availability, which highlights our key motivation-a direct control of the tree-constrained graph generation.

## B. Dataset Details

We describe the details of the datasets used in our experiment.

Synthetic tree pattern dataset To prepare the synthetic dataset, we implement a generator of two-dimensional tree patterns based on the L-system [42], a formal language for describing the growth of the structural form. The L-system recursively applies rewriting rules to the current structure to simulate the growth of branching structures.

Figure S1 shows the initial structures and the rewriting rules we used. At the beginning of the tree generation, an initial sequence is randomly chosen from the pre-defined sequences marked with a purple frame in the figure. At each iteration during the tree generation, the leaf edges ("A" in the sequences) are replaced by a randomly chosen pattern from eight pre-defined ones. A simple example is shown in Fig. S2. We iterate the rewriting process a maximum of three times to generate a tree pattern. We also add randomness to the branch length and joint angles in our dataset. We randomly choose a branch length of scaling [0.5, 2.5] and joint angles of $\left[10^{\circ}, 35^{\circ}\right]$.

Root dataset For the root dataset, the structure of the early-growing roots of Arabidopsis is manually annotated. The structures are annotated by placing points (i.e., graph nodes) on the root path, where the distance between neighboring points may vary depending on the annotator and the images. We, therefore, resample the graph nodes with the same intervals. Starting from keypoints with the degree $\neq 2$

![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-11.jpg?height=351&width=841&top_left_y=238&top_left_x=1092)
Figure S2. An example of the rewriting process. Suppose the initial structure is represented as $\mathbf{F 0 [} \mathbf{+ A 0 ] F 0 [ - A 0 ] A 0}$. If a rewrite rule $\mathbf{F} \rightarrow \mathbf{F} ; \mathbf{A} \rightarrow \mathbf{F}[-\mathbf{A}]$ is applied, i.e., $\mathbf{F}$ remains unchanged and A becomes $\mathbf{F}[-\mathbf{A}]$, the result of the rewrite process is $\mathbf{F 0}[+\mathbf{F 1}[- \mathbf{A 1}] \mathbf{F 0 [}-\mathbf{F 1}[-\mathbf{A 1}] \mathbf{| F 1 [}-\mathbf{A 1}]$. The digits in the sequences indicate the number of times the rewrite is applied.

(i.e., joints and leaf nodes), we sample nodes at intervals of 8 pixels along continuous branch segments.

For data augmentation, we apply flipping, rotation, cropping, noise, lighting, and scaling on the original images. Supposing the roots are almost aligned at seeding, we limit the range of rotation angles in $\left[-9^{\circ},+9^{\circ}\right]$.

Grapevine dataset We use 3D2cut Single Guyot Dataset [18] containing manual annotations on branch structures. We perform data augmentation with rotation angles in $\left[-15^{\circ},+15^{\circ}\right]$ in the same manner as [18]. This dataset also contains the classification of nodes (four classes) and edges (five classes) related to biological meanings. Since the existing two-stage method [18] estimates these categories, we follow the same setup for the two-stage baseline method (refer to the next section for detailed discussions). For the other methods, including our TreeFormer implementation, we use only the binary class information (i.e., branch availability) for generalizability.

## C. Details of Baseline Methods

We describe the implementation details for the baseline methods: The two-stage method and the method with the test-time constraint. Note the implementation for the other baseline, the unconstrained method, is identical to the original RelationFormer [55].

## C.1. Two-stage baseline

Our experiment implements a two-stage baseline involving skeletonization and graph optimization. This baseline implementation is based on ViNet [18], a state-of-the-art plant skeleton estimation method. Since the implementation of [18] is not publicly available, we re-implement the method with reference to the descriptions in the paper. Through the re-implementation, we find room for improvement in the two-stage baseline method. Table S2 compares the performance of our two-stage implementation with a naive re-

Table S2. Quantitative comparisons between our re-implementation of [18] and our two-stage baseline implementation.
| Method | SMD $\downarrow$ | TOPO score $\uparrow$ |  |  | MSE $\downarrow$ |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | Prec. | Rec. | F1 | Node confidence | Edge direction |
| Re-implementation of [18] | $3.84 \times 10^{-3}$ | 0.459 | 0.365 | 0.406 | $6.95 \times 10^{-3}$ | $1.01 \times 10^{-2}$ |
| Our implementation of two-stage method | $\mathbf{4 . 2 4} \times \mathbf{1 0}^{-\mathbf{4}}$ | $\mathbf{0 . 6 7 7}$ | $\mathbf{0 . 5 8 9}$ | $\mathbf{0 . 6 3 0}$ | $\mathbf{1 . 1 9} \times \mathbf{1 0}^{-\mathbf{3}}$ | $\mathbf{2 . 6 7} \times \mathbf{1 0}^{-\mathbf{3}}$ |


![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-12.jpg?height=651&width=1728&top_left_y=523&top_left_x=200)
Figure S3. Network architecture for the first (skeletonization) stage of our two-stage baseline method.

implementation of [18]. The SMD and TOPO scores are the same metrics used in the main paper, and we also compare the mean squared error (MSE) of the first-stage output of both methods. Our implementation achieves a better performance; thus, we use the improved version for our experiment. In the following, we describe the implementation details.

First stage: Skeletonization Similar to ViNet [18], the first stage of our implementation outputs the prediction of node and edge positions as single-channel confidence maps and two-channel vector fields (hereafter referred to as node confidence maps and edge direction maps, respectively). This step is similar to a widespread human pose estimation method i.e., OpenPose [11], which jointly estimates the confidence of person keypoints and the Part Affinity Fields (i.e., two-channel vector fields).

While ViNet [18] uses a sequence of residual blocks followed by the Stacked Hourglass Network [49] for this stage, we use a pre-trained ResNet50 [22] for image feature extraction. This is for a fair comparison to our TreeFormer implementation, which also uses ResNet50 as the backbone ${ }^{9}$. We implement an architecture like the Feature Pyramid Network (FPN) [41], illustrated in Fig. S3, to decode the node \& edge maps from the image features. Figure S4 visually

[^5]compares the estimated node \& edge maps, showing a better accuracy by our two-stage implementation.

The original ViNet estimates multiple classes of nodes (four classes) and edges (five classes) as different maps for the grapevine dataset. Compared to just using binary classes (i.e., a branch exists or not), our two-stage implementation also yields better estimation accuracies using the multiple classes (SMD in $4.2 \times 10^{-4}$ with multi-class and $1.4 \times 10^{-2}$ with binary classes). Therefore, we use the multi-class setup for our two-stage implementation of the grapevine dataset. For the other dataset, we use binary classification since we do not have specific class information.

Second stage: Graph algorithm Given the node confidence and edge direction maps, ViNet [18] first extracts the node positions, followed by the computation of the resistivity between each node pair, defined using the edge directions and the Euclidean distance between nodes. The final estimates of the graph structure are generated using the Dijkstra algorithm, where the tree structure is obtained by computing the shortest paths from all nodes to the detected root crown. The resistivity is used as the edge cost for the Dijkstra algorithm.

For the second stage, we follow the method in [18] except for the graph algorithm used; namely, we compute MST instead of the shortest paths given by the Dijkstra algorithm, since using MST reduces the SMD metric to $4.2 \times 10^{-4}$, compared to $5.9 \times 10^{-4}$ using the Dijkstra al-

![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-13.jpg?height=909&width=1733&top_left_y=243&top_left_x=195)
Figure S4. Visual comparisons between a re-implementation of [18] and our two-stage baseline implementation. Our implementation yields better node confidence and edge direction maps, which are the outputs of the first stage of these methods.

Table S3. Parameters used for the two-stage baseline method. $d$ denotes the distance threshold for the local maximum value search (i.e., non-maximum suppression) of node candidates. $\tau_{m}$ and $\tau_{n}$ are used as the thresholds for node detection from the confidence maps. For the detailed definitions, refer to the original paper [18].
| Dataset | Image size (W, H) [px] | Map size (W, H) [px] | Node confidence diameter [px] | Edge direction width [px] | Node search distance $d[\mathrm{px}]$ | Thresholds for node detection $\tau_{m}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Synthetic | 512, 512 | 512, 512 | 4 | 5 | 9 | 0.97 |
| Root | 570, 190 | 570, 190 | 3 | 5 | 7 | 0.99 |
| Grapevine | 1008, 756 | 256, 256 | 3 | 10 | 25 | 0.97 |


gorithm for the grapevine dataset.

Detailed parameter settings The two-stage method involves heuristic parameters for node and edge detection. Therefore, we empirically select the best parameter sets for each dataset. Table S3 lists the detailed parameters. In particular, for the root dataset, we need to carefully tune some hyperparameters (namely, $d, \tau_{m}$, and $\tau_{n}$ in the table) to yield reasonable estimates, where the configurations yielding the best SMD scores are reported in the main paper.

## C.2. Test-time constraint baseline

For the test-time constraint baseline, we apply MST only in the inference phase, where the graph generator is trained using the same procedure as the unconstrained method. The MST used in this baseline method is identical to our proposed one.

## D. Performance Analysis

In this section, we present a detailed analysis of the performance of our proposed method. The effectiveness of our method is evaluated through comprehensive experiments in different scenarios. Specifically, we compare our method with the Auto-regressive (AR) model, and we also analyze the performance when our method is applied solely during the training processes.

## D.1. Comparison with auto-regressive (AR) method

While we implement the tree-graph constraint on the state-of-the-art non-autoregressive graph generator, RelationFormer [55], other choices of constrained graph generation are viable. Existing works aiming for tree-constrained graph generation, such as in molecule structure estimation [4, 27, 28], use auto-regressive (AR) graph generation. AR methods are a simpler choice for imposing the

Table S4. Comparisons of different graph generation models (i.e., RelationFormer [55] and GGT [7]) on the synthetic dataset.
| Method | SMD $\downarrow$ | TOPO score $\uparrow$ |  |  | Tree rate |
| :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | Prec. | Rec. | F1 | $[\%]$ |
| GGT [7] | $2.71 \times 10^{-3}$ | 0.635 | 0.537 | 0.582 | 92.06 |
| GGT w/ test-time constraint | $4.13 \times 10^{-3}$ | 0.620 | 0.545 | 0.580 | 98.10 |
| GGT w/ SFS layer | $2.80 \times 10^{-3}$ | 0.652 | 0.584 | 0.616 | 99.63 |
| RelationFormer [55] w/ SFS layer (Ours) | $4.78 \times 10^{-6}$ | 0.986 | 0.968 | 0.977 | 100.0 |


constraint since it is relatively straightforward to implement the tree-graph constraint in their graph development process. However, since the AR methods generate graph nodes and edges progressively, they are prone to breakdowns due to changes in the output order or errors during the generation. This tendency is particularly pronounced for relatively large graphs, including our setup.

To assess the potential of AR methods, we test the state-of-the-art transformer-based AR graph generator, Generative Graph Transformer (GGT) [7]. Table S4 compares our method and several variances of GGT on the synthetic tree pattern dataset. The results show that the accuracy of GGT falls short compared to our method, although the vanilla GGT (the top row) mostly outputs tree graphs ( $92 \%$ ) without explicitly imposing the tree-graph constraint. We identified that errors by GGT occurring at a particular step in the AR generation process continuously cause errors in the sequence of following generations. The GGT was initially designed for small datasets, specifically for graphs with $|V| \leq 10$. For our setup, where $|V| \geq 100$, generating these long sequences in a specific order presents a significant challenge.

## D.2. Effectiveness of tree-constraint during training

To assess whether our SFS layer (positively) affects the training process itself or not, we evaluate our method without using the SFS layer and the MST algorithm during the inference phase, i.e., introducing constraint only during the training phase (called train-time constraint hereafter). Table S5 summarizes the performances. Inducting the tree constraint during the training phase mostly outperforms the methods without constraints, meaning that the improvement by our method is based on network improvement by the loss propagated via the SFS layer. We also checked the change in the accuracy metric during training, and found our method consistently achieved better accuracy from the beginning of the training.

## E. Other Design Choices

The experiments in the main paper already provide some ablation studies, namely, comparisons of our method with 1) graph generation without constraint (unconstrained), and 2) a method without using MST in the training loop (testtime constraint). Here, we delve further into the potential design choices of our TreeFormer model.

Table S5. Quantitative results with additional baseline, train-time constraint.
| Dataset | Method | SFS | SMD $\downarrow$ | TOPO score $\uparrow$ |  |  | Tree rate [\%] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  |  |  |  | Prec. | Rec. | F1 |  |
| Synthetic | Unconstrained |  | $1.43 \times 10^{-5}$ | 0.978 | 0.929 | 0.953 | 36.2 |
|  | Test-time constraint |  | $6.26 \times 10^{-6}$ | 0.977 | 0.953 | 0.965 | 100.0 |
|  | Train-time constraint | $\checkmark$ | $8.44 \times 10^{-6}$ | 0.987 | 0.954 | 0.970 | 56.5 |
|  | Ours | $\checkmark$ | $4.78 \times 10^{-6}$ | 0.986 | 0.968 | 0.977 | 100.0 |
| Root | Unconstrained |  | $1.19 \times 10^{-4}$ | 0.831 | 0.633 | 0.719 | 35.9 |
|  | Test-time constraint |  | $1.52 \times 10^{-4}$ | 0.829 | 0.771 | 0.799 | 100.0 |
|  | Train-time constraint | $\checkmark$ | $7.81 \times 10^{-5}$ | 0.853 | 0.619 | 0.718 | 37.2 |
|  | Ours | $\checkmark$ | $8.82 \times 10^{-5}$ | 0.861 | 0.807 | 0.833 | 100.0 |
| Grapevine | Unconstrained |  | $1.45 \times 10^{-4}$ | 0.963 | 0.559 | 0.708 | 0.0 |
|  | Test-time constraint |  | $1.47 \times 10^{-4}$ | 0.896 | 0.840 | 0.867 | 100.0 |
|  | Train-time constraint | $\checkmark$ | $1.30 \times 10^{-4}$ | 0.965 | 0.566 | 0.713 | 0.0 |
|  | Ours | $\checkmark$ | $1.03 \times 10^{-4}$ | 0.899 | 0.843 | 0.870 | 100.0 |


## E.1. Other graph generators

Although the proposed module, the SFS layer, can be easily integrated into graph generators other than RelationFormer [55], we found that no methods but our TreeFormer implementation achieve satisfactory results. Here, we discuss results by the implementation of our method to the AR graph generator, GGT [7], which achieves the second-best accuracy for multiple datasets following the state-of-the-art RelationFormer.

Table S4 in the last section compares the GGT with and without the tree-graph constraint. Compared to the GGT with test-time MST, using our SFS layer on top of GGT improves both SMD and TOPO scores ${ }^{10}$, although the accuracies are insufficient in practice due to the drawback of AR-based generation processes discussed above. Using the newer RelationFormer model significantly improves the estimation accuracy, which implies that our SFS layer will benefit from the future development of graph generation models.

## E.2. Using node distances for edge cost in MST

Although our proposed method uses the edge non-existence probabilities $\left\{\hat{y}_{(i, j)}^{-}\right\}$as the edge cost for the MST algorithm, inspired by the two-stage method that uses node distance for the edge cost computation, we multiply the Euclidean distance between nodes by our original edge cost.

As a result, SMD with modified edge cost does not improve accuracy (it achieves the same SMD as our method in the Grapevine dataset). A possible reason for this is that the graph generator itself can take the node distance into account when estimating graph edges. Therefore, we simply use the edge non-existence probabilities $\left\{\hat{y}_{(i, j)}^{-}\right\}$as the edge cost for our method.

## E.3. Ablation for $\Lambda$

An important hyperparameter in our method is $\Lambda$, which controls the level of suppression for unwanted features.

[^6]Table S6. Ablation for $\Lambda$.
| $\Lambda$ | SMD $\downarrow$ | TOPO score $\uparrow$ |  |  |
| :---: | :---: | :---: | :---: | :---: |
|  | Prec. | Rec. | F1 |  |
| $2\left(\exp (-\Lambda)=1.4 \times 10^{-1}\right)$ | $1.51 \times 10^{-4}$ | 0.871 | 0.803 | 0.836 |
| $5\left(\exp (-\Lambda)=6.7 \times 10^{-3}\right)$ | $1.27 \times 10^{-4}$ | 0.866 | 0.799 | 0.831 |
| $10\left(\exp (-\Lambda)=4.5 \times 10^{-5}\right)$ | $\mathbf{1 . 0 3} \times \mathbf{1 0}^{-\mathbf{4}}$ | $\mathbf{0 . 8 9 9}$ | $\mathbf{0 . 8 4 3}$ | $\mathbf{0 . 8 7 0}$ |
| $100\left(\exp (-\Lambda)=3.7 \times 10^{-44}\right)$ | $1.07 \times 10^{-4}$ | 0.886 | 0.830 | 0.857 |


Here, we report an ablation study for this parameter using the grapevine dataset. Table S6 shows that our choice ( $\Lambda=10$ ) achieves better, while the changes in $\Lambda$ do not significantly affect the overall accuracy as long as $\exp (-\Lambda)$ is close enough to zero. This result indicates that our method is robust to the hyperparameter setting.

## F. Additional Visual Results

We finally show additional visual results. Figures S5 and S6 show the additional results for synthetic and root datasets, respectively. Figures S7 and S8 show the results for the grapevine dataset. Figure S9 show the results for out-ofdomain testing.

These results consistently demonstrate the high-fidelity estimation of plant skeletons by our TreeFormer, which uses the SFS layer that incorporates the constraints while training graph generation models.

## References

[1] Sherif Abdelkarim, Aniket Agarwal, Panos Achlioptas, Jun Chen, Jiaji Huang, Boyang Li, Kenneth Ward Church, and Mohamed Elhoseiny. Exploring long tail visual relationship recognition with large vocabulary. In Proceedings of IEEE/CVF International Conference on Computer Vision (ICCV), pages 15901-15910, 2020. 2
[2] David Acuna, Huan Ling, Amlan Kar, and Sanja Fidler. Efficient interactive annotation of segmentation datasets with polygon-RNN++. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 859-868, 2018. 1, 2
[3] Akshay Agrawal, Brandon Amos, Shane Barratt, Stephen Boyd, Steven Diamond, and J Zico Kolter. Differentiable convex optimization layers. In Advances in Neural Information Processing Systems (NeurIPS), volume 32, 2019. 3
[4] Sungsoo Ahn, Binghong Chen, Tianzhe Wang, and Le Song. Spanning tree-based graph generation for molecules. In Proceedings of International Conference on Learning Representations (ICLR), 2022. 3, 13
[5] Mingyao Ai, Yuan Yao, Qingwu Hu, Yue Wang, and Wei Wang. An automatic tree skeleton extraction approach based on multi-view slicing using terrestrial LiDAR scans data. Remote Sensing, 12(22), 2020. 2
[6] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016. 5
[7] Davide Belli and Thomas Kipf. Image-conditioned graph generation for road network extraction. In Proceedings
of NeurIPS Workshop on Graph Representation Learning, 2019. 1, 2, 6, 14
[8] Simon Bohlender, Ilkay Öksüz, and A. Mukhopadhyay. A survey on shape-constraint deep learning for medical image segmentation. IEEE Reviews in Biomedical Engineering, 16:225-240, 2021. 3
[9] Alexander Bucksch. A practical introduction to skeletons for the plant sciences. Applications in Plant Sciences, 2(8):1400005, 2014. 2
[10] Llorenç Cabrera-Bosquet, Christian Fournier, Nicolas Brichet, Claude Welcker, Benoît Suard, and François Tardieu. High-throughput estimation of incident light, light interception and radiation-use efficiency of thousands of plants in a phenotyping platform. New Phytologist, 212(1):269-281, 2016. 1
[11] Zhe Cao, Tomas Simon, Shih-En Wei, and Yaser Sheikh. Realtime multi-person 2D pose estimation using part affinity fields. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 1, 2, 8, 12
[12] Ayan Chaudhury and Christophe Godin. Skeletonization of plant point cloud data using stochastic optimization framework. Frontiers in Plant Science, 11, 2020. 2
[13] Meng-Jiun Chiou, Henghui Ding, Hanshu Yan, Changhu Wang, Roger Zimmermann, and Jiashi Feng. Recovering the unbiased scene graphs from the biased ones. In Proceedings of ACM International Conference on Multimedia (MM), pages 1581-1590, 2021. 2
[14] Yuren Cong, Michael Ying Yang, and Bodo Rosenhahn. RelTR: Relation transformer for scene graph generation. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 45(9):11169-11183, 2023. 2
[15] Shenglan Du, Roderik Lindenbergh, Hugo Ledoux, Jantien Stoter, and Liangliang Nan. AdTree: Accurate, detailed, and automatic modelling of laser-scanned trees. Remote Sensing, 11(18), 2019. 2
[16] Aaron Ferber, Bryan Wilder, Bistra Dilkina, and Milind Tambe. Mipaal: Mixed integer program as a layer. In Proceedings of AAAI Conference on Artificial Intelligence (AAAI), pages 1504-1511, 2020. 3
[17] Mathieu Gaillard, Chenyong Miao, James c. Schnable, and Bedrich Benes. Sorghum segmentation by skeleton extraction. In Proceedings of European Conference on Computer Vision (ECCV) Workshops, 2020. 1
[18] Theophile Gentilhomme, Michael Villamizar, Jerome Corre, and Jean-Marc Odobez. Towards smart pruning: ViNet, a deep-learning approach for grapevine structure estimation. Computers and Electronics in Agriculture, 207:107736, 2023. 1, 2, 6, 7, 11, 12, 13
[19] Valerio Giuffrida, Massimo Minervini, and Sotirios Tsaftaris. Learning to count leaves in rosette plants. In Proceedings of Workshop on Computer Vision Problems in Plant Phenotyping (CVPPP), pages 1.1-1.13, 01 2015. 2
[20] Jianwei Guo, Haiyong Jiang, Bedrich Benes, Oliver Deussen, Xiaopeng Zhang, Dani Lischinski, and Hui Huang. Inverse procedural modeling of branching structures by inferring L-systems. ACM Transactions on Graphics (TOG), 39(5):1-13, 2020. 6
[21] Harshit Gupta, Kyong Hwan Jin, Ha Q Nguyen, Michael T

| Ground truth | Two-stage | Unconstrained | Test-time constraint | Ours |
| :--- | :--- | :--- | :--- | :--- |
| ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=253&width=237&top_left_y=311&top_left_x=373) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=255&width=233&top_left_y=309&top_left_x=674) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=255&width=233&top_left_y=309&top_left_x=973) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=255&width=240&top_left_y=309&top_left_x=1272) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=255&width=239&top_left_y=309&top_left_x=1574) |
| ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=257&width=225&top_left_y=606&top_left_x=381) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=259&width=231&top_left_y=604&top_left_x=676) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=259&width=228&top_left_y=604&top_left_x=978) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=259&width=233&top_left_y=604&top_left_x=1279) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=259&width=233&top_left_y=604&top_left_x=1578) |
| ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=241&width=218&top_left_y=921&top_left_x=364) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=244&width=224&top_left_y=921&top_left_x=663) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=244&width=223&top_left_y=921&top_left_x=962) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=244&width=222&top_left_y=921&top_left_x=1264) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=244&width=224&top_left_y=921&top_left_x=1561) |
| ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=246&width=290&top_left_y=1216&top_left_x=316) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=246&width=290&top_left_y=1220&top_left_x=615) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=242&width=289&top_left_y=1220&top_left_x=917) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=242&width=294&top_left_y=1220&top_left_x=1216) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=246&width=292&top_left_y=1220&top_left_x=1517) |
| ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=246&width=194&top_left_y=1515&top_left_x=416) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=246&width=190&top_left_y=1515&top_left_x=719) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=246&width=189&top_left_y=1515&top_left_x=1019) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=246&width=194&top_left_y=1515&top_left_x=1318) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=246&width=192&top_left_y=1515&top_left_x=1619) |
| ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=237&width=218&top_left_y=1823&top_left_x=344) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=233&width=225&top_left_y=1827&top_left_x=641) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=233&width=220&top_left_y=1827&top_left_x=943) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=233&width=220&top_left_y=1827&top_left_x=1244) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=233&width=220&top_left_y=1827&top_left_x=1543) |
| ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=220&width=194&top_left_y=2139&top_left_x=412) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=220&width=192&top_left_y=2139&top_left_x=715) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=220&width=189&top_left_y=2139&top_left_x=1019) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=220&width=194&top_left_y=2139&top_left_x=1314) | ![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-16.jpg?height=220&width=196&top_left_y=2139&top_left_x=1615) |

Figure S5. Additional results for the synthetic branch pattern dataset.

![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-17.jpg?height=2151&width=1554&top_left_y=246&top_left_x=284)
Figure S6. Additional results for the root dataset.

![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-18.jpg?height=2145&width=1408&top_left_y=249&top_left_x=357)
Figure S7. Additional results for the grapevine dataset.

![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-19.jpg?height=2145&width=1408&top_left_y=249&top_left_x=357)
Figure S8. Additional results for the grapevine dataset (cont'd).

![](https://cdn.mathpix.com/cropped/2025_11_14_299ac0df8c4c85f4e4a5g-20.jpg?height=1044&width=1725&top_left_y=249&top_left_x=200)
Figure S9. Additional results for the out-of-domain test dataset.

McCann, and Michael Unser. CNN-based projected gradient descent for consistent CT image reconstruction. IEEE Transactions on Medical Imaging, 37(6):1440-1453, 2018. 3
[22] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 770-778, 2016. 12
[23] Songtao He, Favyen Bastani, Sofiane Abbar, Mohammad Alizadeh, Hari Balakrishnan, Sanjay Chawla, and Sam Madden. RoadRunner: Improving the precision of road network inference from gps trajectories. In Proceedings of ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, pages 3-12, 2018. 6
[24] Songtao He, Favyen Bastani, Satvat Jagwani, Mohammad Alizadeh, Hari Balakrishnan, Sanjay Chawla, Mohamed M. Elshrif, Samuel Madden, and Mohammad Amin Sadeghi. Sat2Graph: Road graph extraction through graph-tensor encoding. In Proceedings of European Conference on Computer Vision (ECCV), pages 51-67, 2020. 1, 2, 6
[25] Hui Huang, Shihao Wu, Daniel Cohen-Or, Minglun Gong, Hao Zhang, Guiqing Li, and Baoquan Chen. L1-medial skeleton of point cloud. ACM Transactions on Graphics (TOG), 32(4):65, 2013. 2
[26] Takahiro Isokane, Fumio Okura, Ayaka Ide, Yasuyuki Matsushita, and Yasushi Yagi. Probabilistic plant modeling via multi-view image-to-image translation. In Proceedings
of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2906-2915, 2018. 2
[27] Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Junction tree variational autoencoder for molecular graph generation. In Proceedings of International Conference on Machine Learning (ICML), pages 2323-2332, 2018. 3, 13
[28] Wengong Jin, Regina Barzilay, and T. Jaakkola. Hierarchical generation of molecular graphs using structural motifs. In Proceedings of International Conference on Machine Learning (ICML), 2020. 3, 13
[29] Ivan Khokhlov, Lev Krasnov, Maxim V. Fedorov, and Sergey Sosnin. Image2SMILES: Transformer-based molecular optical recognition engine. Chemistry-Methods, 2(1):e202100069, 2022. 1, 2
[30] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In Yoshua Bengio and Yann LeCun, editors, Proceedings of International Conference on Learning Representations (ICLR), 2014. 2
[31] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In Proceedings of International Conference on Learning Representations (ICLR), 2014. 3
[32] Thomas N Kipf and Max Welling. Variational graph autoencoders. In Proceedings on NeurIPS Workshop on Bayesian Deep Learning, 2016. 2
[33] James Kotary, Ferdinando Fioretto, Pascal Van Hentenryck, and Bryan Wilder. End-to-end constrained optimization learning: A survey. In Proceedings of International Joint Conferences on Artificial Intelligence (IJCAI), 2021. 3
[34] Joseph B Kruskal. On the shortest spanning subtree of a graph and the traveling salesman problem. Proceedings of the American Mathematical Society, 7(1):48-50, 1956. 5
[35] Rongjie Li, Songyang Zhang, and Xuming He. SGTR: End-to-end scene graph generation with transformer. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19486-19496, 2022. 1, 2
[36] Weijian Li, Yuhang Lu, Kang Zheng, Haofu Liao, Chihung Lin, Jiebo Luo, Chi-Tung Cheng, Jing Xiao, Le Lu, ChangFu Kuo, et al. Structured landmark detection via topologyadapting deep graph learning. In Proceedings of European Conference on Computer Vision (ECCV), pages 266-283, 2020. 1
[37] Weijia Li, Wenqian Zhao, Huaping Zhong, Conghui He, and Dahua Lin. Joint semantic-geometric learning for polygonal building segmentation. In Proceedings of AAAI Conference on Artificial Intelligence (AAAI), pages 1958-1965, 2021. 1, 2
[38] Yuanwei Li, Chin Pang Ho, Matthieu Toulemonde, Navtej Chahal, Roxy Senior, and Meng-Xing Tang. Fully automatic myocardial segmentation of contrast echocardiography sequence using random forests guided by shape model. IEEE Transactions on Medical Imaging, 37(5):1081-1091, 2017. 3
[39] Yujia Li, Oriol Vinyals, Chris Dyer, Razvan Pascanu, and Peter Battaglia. Learning deep generative models of graphs. arXiv preprint arXiv:1803.03324, 2018. 2
[40] Justin Liang, Namdar Homayounfar, Wei-Chiu Ma, Yuwen Xiong, Rui Hu, and Raquel Urtasun. PolyTransform: Deep polygon transformer for instance segmentation. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9128-9137, 2020. 1, 2
[41] Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2117-2125, 2017. 12
[42] Aristid Lindenmayer. Mathematical models for cellular interactions in development I. Filaments with one-sided inputs. Journal of Theoretical Biology, 18(3):280-299, 1968. 6, 11
[43] Huan Ling, Jun Gao, Amlan Kar, Wenzheng Chen, and Sanja Fidler. Fast interactive object annotation with curve-GCN. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5252-5261, 2019. 1, 2
[44] Qi Liu, Miltiadis Allamanis, Marc Brockschmidt, and Alexander L. Gaunt. Constrained graph variational autoencoders for molecule design. In Advances in Neural Information Processing Systems (NeurIPS), 2018. 2
[45] Youzhi Luo, Keqiang Yan, and Shuiwang Ji. Graphdf: A discrete flow model for molecular graph generation. In Proceedings of International Conference on Machine Learning (ICML), 2021. 2
[46] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In Proceedings of International Conference on Learning Representations (ICLR), 2018. 3
[47] Teng Miao, Chao Zhu, Tongyu Xu, Tao Yang, Na Li,

Yuncheng Zhou, and Hanbing Deng. Automatic stemleaf segmentation of maize shoots using three-dimensional point cloud. Computers and Electronics in Agriculture, 187:106310, 2021. 1, 2
[48] Young-Soo Myung, Chang-Ho Lee, and Dong-Wan Tcha. On the generalized minimum spanning tree problem. Networks, 26(4):231-241, 1995. 3
[49] Alejandro Newell, Kaiyu Yang, and Jia Deng. Stacked hourglass networks for human pose estimation. In Proceedings of European Conference on Computer Vision (ECCV), pages 11-16, 2016. 12
[50] Fumio Okura. 3D modeling and reconstruction of plants and trees: A cross-cutting review across computer graphics, vision, and plant phenotyping. Breeding Science, 72(1):31-47, 2022. 2
[51] Jeffrey Pennington, Richard Socher, and Christopher D Manning. Glove: Global vectors for word representation. In Proceedings of Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1532-1543, 2014. 2
[52] Petrică C Pop. The generalized minimum spanning tree problem: An overview of formulations, solution procedures and latest advances. European Journal of Operational Research, 283(1):1-15, 2020. 3
[53] Sahand Sharifzadeh, Sina Moayed Baharlou, Martin Schmitt, Hinrich Schütze, and Volker Tresp. Improving scene graph classification by exploiting knowledge from texts. In Proceedings of AAAI Conference on Artificial Intelligence (AAAI), pages 2189-2197, 2022. 2
[54] wu Sheng, Weiliang Wen, Boxiang Xiao, Xinyu Guo, Jian Jun Du, Chuanyu Wang, and Yongjian Wang. An accurate skeleton extraction approach from 3d point clouds of maize plants. Frontiers in Plant Science, 10:248, 2019. 1, 2
[55] Suprosanna Shit, Rajat Koner, Bastian Wittmann, Johannes Paetzold, Ivan Ezhov, Hongwei Li, Jiazhen Pan, Sahand Sharifzadeh, Georgios Kaissis, Volker Tresp, et al. Relationformer: A unified framework for image-to-graph generation. In Proceedings of European Conference on Computer Vision (ECCV), pages 422-439, 2022. 1, 2, 5, 6, 7, 11, 13, 14
[56] Vibhav Vineet, Pawan Harish, Suryakant Patidar, and PJ Narayanan. Fast minimum spanning tree for large graphs on the GPU. In Proceedings of Conference on High Performance Graphics (HPG), pages 167-171, 2009. 8
[57] Bryan Wilder, Bistra Dilkina, and Milind Tambe. Melding the data-decisions pipeline: Decision-focused learning for combinatorial optimization. In Proceedings of AAAI Conference on Artificial Intelligence (AAAI), pages 1658-1665, 2019. 3
[58] Sheng Wu, Weiliang Wen, Yongjian Wang, Jiangchuan Fan, Chuanyu Wang, Wenbo Gou, and Xinyu Guo. MVS-Pheno: A portable and low-cost phenotyping platform for maize shoots using multiview stereo 3D reconstruction. Plant Phenomics, 2020:1848437, 2020. 2
[59] Cihang Xie, Mingxing Tan, Boqing Gong, Jiang Wang, Alan L. Yuille, and Quoc V. Le. Adversarial examples improve image recognition. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 3
[60] Hui Xu, Nathan Gossett, and Baoquan Chen. Knowledge and heuristic-based modeling of laser-scanned trees. ACM

Transactions on Graphics (TOG), 26(4):19, 2007. 2
[61] Zhenhua Xu, Yuxuan Liu, Yuxiang Sun, Ming Liu, and Lujia Wang. RNGDet++: Road network graph detection by transformer with instance segmentation and multi-scale features enhancement. IEEE Robotics and Automation Letters, pages 1-8, 2023. 1, 2
[62] Zhenhua Xu, Yuxiang Sun, and Ming Liu. iCurb: Imitation learning-based detection of road curbs using aerial images for autonomous driving. IEEE Robotics and Automation Letters, 6:1097-1104, 2021. 1, 2
[63] Jingkang Yang, Yi Zhe Ang, Zujin Guo, Kaiyang Zhou, Wayne Zhang, and Ziwei Liu. Panoptic scene graph generation. In Proceedings of European Conference on Computer Vision (ECCV), pages 178-196, 2022. 2
[64] Michitaka Yoshida, Akihiko Torii, Masatoshi Okutomi, Kenta Endo, Yukinobu Sugiyama, Rin-ichiro Taniguchi, and Hajime Nagahara. Joint optimization for compressive video sensing and reconstruction under hardware constraints. In Proceedings of European Conference on Computer Vision (ECCV), pages 634-649, 2018. 3
[65] Chengxi Zang and Fei Wang. Moflow: A invertible flow model for generating molecular graphs. In Proceedings of ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining (KDD), pages 617-626, 2020. 2
[66] Xiaochen Zhou, Bosheng Li, Bedrich Benes, Songlin Fei, and Sören Pirk. Deeptree: Modeling trees with situated latents. IEEE Transactions on Visualization and Computer Graphics, 30(8):2795-5809, 2023. 2
[67] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable DETR: Deformable transformers for end-to-end object detection. In Proceedings of International Conference on Learning Representations (ICLR), 2021. 5, 12
[68] Illia Ziamtsov and Saket Navlakha. Machine learning approaches to improve three basic plant phenotyping tasks using three-dimensional point clouds. Plant Physiology, 181(4):1425-1440, 2019. 2


[^0]:    ${ }^{1}$ See the supplementary materials for the derivation.
    ${ }^{2}$ This is akin to the dropout layer often used in neural networks.

[^1]:    ${ }^{3}$ We omit the subscript $(i, j)$ for simplicity.
    ${ }^{4}$ See supplementary materials for the mathematical proof.

[^2]:    ${ }^{5}$ Denoted as $\mathcal{L}_{\text {rln }}$ in the original paper [55], we use $\mathcal{L}_{\text {edge }}$ for generality.
    ${ }^{6}$ https: / /networkx.org/, last accessed on July 15, 2024.

[^3]:    7https: / / github. com / suprosanna / relationformer, last accessed on July 15, 2024.

[^4]:    ${ }^{8}$ We omit the subscript $(i, j)$ for simplicity.

[^5]:    ${ }^{9}$ ResNet50 is actually used as the node detection module in RelationFormer (that is based on Deformable DETR [67]), which is the basis of our TreeFormer, and we inherited its implementation.

[^6]:    ${ }^{10}$ GGT w/ SFS layer does not achieve 100 [\%] tree rate because it sometimes fails to generate any graphs.

