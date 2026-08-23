## Generative Manifold Networks (GMN)
---
Generative Manifold Networks is a generalization of nonlinear dynamical systems from a single state-space with a manifold operator, to an interconnected network of operators on the state-space(s) see: [Park et al.](https://doi.org/10.64898/2026.05.12.724527)

GMN is developed at the [Biological Nonlinear Dynamics Data Science Unit, OIST](https://www.oist.jp/research/research-units/bndd)

---
## Installation

### Python Package Index (PyPI) [gmn](https://pypi.org/project/gmn/). 

`pip install gmn`

---
## Documentation
[GMN documentation](https://pao-unit.github.io/gmn/).

---
## Usage
Example usage at the python prompt in directory `gmn/config`:
```python
>>> import gmn
>>> G = gmn.GMN( configFile = './default.cfg' )
>>> G.Generate()
>>> G.DataOut.tail()
     Time       A       C       D         B       Out
295   996 -0.2487 -0.5018  0.7500  0.985236 -0.979370
296   997 -0.1874 -0.4708  0.7937  0.985842 -0.991504
297   998 -0.1253 -0.4248  0.8177  0.965066 -0.973041
298   999 -0.0628 -0.3671  0.8224  0.923630 -0.931681
299  1000  0.0000 -0.3016  0.8090  0.862222 -0.871642
```

---
### References
[Experimentally testable whole brain manifolds that recapitulate behavior](https://arxiv.org/abs/2106.10627)
