## Generative Manifold Networks (GMN)
---
Generative Manifold Networks is a generalization of nonlinear dynamical systems from a single state-space with a manifold operator, to an interconnected network of operators on the state-space(s) introduced by [Pao et al.](https://arxiv.org/abs/2106.10627)

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
Example usage at the python prompt in directory `gmn`:
```python
>>> import gmn
>>> G = gmn.GMN( configFile = './config/default.cfg' )
>>> G.Generate()
>>> G.DataOut.tail()
     Time             A         B       C         D       Out
295   996 -2.487000e-01  0.927389 -0.5018  0.383759 -0.902106
296   997 -1.874000e-01  0.973968 -0.4708  0.471114 -0.961839
297   998 -1.253000e-01  0.989932 -0.4248  0.540129 -0.989022
298   999 -6.280002e-02  0.984369 -0.3671  0.591274 -0.986631
299  1000 -2.438686e-08  0.957464 -0.3016  0.624011 -0.951023
```

---
### References
[Experimentally testable whole brain manifolds that recapitulate behavior](https://arxiv.org/abs/2106.10627)
