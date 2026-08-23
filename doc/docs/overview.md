## GMN Overview
---
Network structure is a foundation of complex systems as demonstrated in genomic ([Turner 2014](https://doi.org/10.1371/journal.pgen.1004162)), metabolic ([Jeong 2001](https://doi.org/10.1038/35075138)), physiologic ([Bashan 2012](https://doi.org/10.1038/ncomms1705)), social, and neural systems/networks ([Bae 2025](https://doi.org/10.1038/s41586-025-08790-w),[Assaf 2020](https://doi.org/10.1038/s41593-020-0641-7)). Networks often express low-dimensional structure within a high-dimensional system consistent with the manifold hypothesis ([Thibeault 2024](https://doi.org/10.1038/s41567-023-02303-0)) evidenced in living systems ([Eckmann 2021](https://onlinelibrary.wiley.com/doi/abs/10.1002/bies.202100062)) and their neural processing ([Fontenele 2024](https://www.science.org/doi/abs/10.1126/sciadv.adj9303)). The fact that fantastically complex structures such as mammalian brains express function and behavior not as single, ultra high-dimensional objects, but as interacting networks suggests the computational architecture should encompass low-dimensional, multiscale, interacting networks.

Generative manifold networks (GMN) combine these essential features into a new architecture based on interactive dynamical manifolds. GMN networks are discovered through an interaction function between observables defining an adjacency matrix from which a network graph for a desired target observable is grown. The interaction function can be a metric of causality such as convergent cross mapping (CCM) ([Sugihara 2012](https://www.science.org/doi/10.1126/science.1227079)), mutual information, nonlinearity ([Sugihara 1994](https://royalsocietypublishing.org/doi/10.1098/rsta.1994.0106),[Smith 2015](https://onlinelibrary.wiley.com/doi/abs/10.1002/sta4.96),[Pao 2021](https://arxiv.org/abs/2106.10627),) or other suitable interaction metric. Each node of the network is a multivariate state space manifold driven by observables leveraging the power of generalized embedding ([Deyle 2011](https://www.ncbi.nlm.nih.gov/pubmed/21483839)). The architecture is therefore simple, low-dimensional and observable.

GMN can be used to discover manifold networks underlying complex, nonlinear dynamical systems toward understanding and prediction of their internal states and behaviors ([Park et al. 2026](https://doi.org/10.64898/2026.05.12.724527)).

### Manifold Representation
Many state-space analytic methods presume the state-space can be represented as an invariant manifold completely encapsulating the system dynamics. Another common presumption is that a diffeomorphic representation of the true manifold can be derived from Takens embedding of a _univariate_ time series observed from the system.  In practice, univariate observations may not sufficiently encapsulate the underlying dynamics, an issue that is compounded as dimensionaliy and complexity of the dynamics increase.

In cases where the dynamics arise from a complex interaction network, the realities of partial observation imply that univariate reconstructions from various time series of the system may not provide a diffeomorphic representation of a global underlying manifold. Instead, decomposing the system into an interacting network of manifolds may better represent the underlying dynamics.

### Network Determination
Manifold network structure is discovered using an interaction matrix quantifying  interactions between all observables.

#### Interaction Matrix
Given a data set with N observation vectors the interaction matrix (iMatrix) is an NxN matrix with each entry quantified by application of an interaction function F() between all combination of observables.

---
![alt text](imgs/InteractionMatrix_I.png "Interaction Matrix")
---

The application program [`InteractionMatrix.py`](https://github.com/NonlinearDynamicsDSU/gmn/blob/c770cc0a83df9fb4cc5f1ed7e555192c80d1c592/apps/InteractionMatrix.py) can create interaction matrices using a variety of interaction functions F().

Available functions include :

| Method                          | Label   | argument |
| ------------------------------- | ------- | -------- |
| Cross Correlation               | CC      | -rho     |
| Simplex Cross Map               | CM      | -cmap    |
| Convergent Cross Map            | CCM     | -ccm     |
| rho Diff = max(CM, 0) - abs(CC) | rhoDiff | -rhoDiff |
| Mutual Information              | MI      | -mi      |
| Mutual Information Non Linearity| MI\_NL  | -nl      |
| SMap nonlinearity               | SMap    | -smap    |
| CCM : Mutual Information        | CMI     | -cmi     |

A natural choice for the interaction function F() is convergent cross mapping (CCM) since CCM validates that the two variables are part of the same dynamical system. Results are stored in a Python pickled dictionary of pandas dataFrames or output as .csv files. See the docstring in [`InteractionMatrix.py`](https://github.com/NonlinearDynamicsDSU/gmn/blob/c770cc0a83df9fb4cc5f1ed7e555192c80d1c592/apps/InteractionMatrix.py).

---

#### Network Creation

Once an interaction matrix is defined the application program `CreateNetwork.py` builds the GMN network as a [`networkx`](https://networkx.org/) acyclic directed graph (DiGraph). Nodes are added recursively starting at the output node(s) adding links according to the network interaction matrix while disallowing creation of network cycles. The network is stored in a Python dictionary with keys `Graph` : the networkx DiGraph, and `Map` : a Python dictionary of node names. 

![alt text](imgs/CreateNetwork_II.png "Create Network")


#### Generative Mode

Each `Node` consists of a low-dimensional state-space manifold with an output corresponding to an observation vector. Given the network and observation vectors a `GMN` class object is created and initialized according to a configuration file.

```python
import gmn
G = gmn.GMN( configFile = 'config/default.cfg' )
```
The network is run in generative mode to generate time series at all nodes:

```python
G.Generate()
```

![alt text](imgs/GenerativeMode_III.png "Generative Mode")
