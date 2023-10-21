# QMpytorch
Use pytorch to get high performance derivatives for quantum mechanical calculation.

In QMpytorch.py you can add a free multiparticle wave function with parameters, which are optimized by minimizing the energy.

The example is a toy system, one dimension 3 nuclei and 3 electrons with an Gauss potential (to make Monte-Carlo VEGAS integration easier).

In the subfolder Enganglement we suggest a spacial entanglement measure, which will be used later.

Install and run
```
conda create --name qmpytorch
conda activate qmpytorch
conda install pytorch
pip install noisyopt
pip install ./torchquad/

python QMpytorch.py 
```