## NN_Comparison | Neural Networks Comparison

### General information

This study project was created in order to demonstrate working results (accuracy, outputs prediction) for
several optimization methods used as training methods for classification task in feedforward neural network.

List of used methods:

  1. **Stochastic gradient descent with momentum** *(SGD with momentum)*:

    M. Buscema, “Back propagation neural networks,” Subst. Use Misuse, vol. 33, no. 2, pp. 233–270, 1998, doi: 10.3109/10826089809115863.

  2. **Particle Swarm Optimization** *(PSO)*:

    R. Eberhart and J. Kennedy, “New optimizer using particle swarm theory,” in Proceedings of the International Symposium on Micro Machine and Human Science, 1995, pp. 39–43, doi: 10.1109/mhs.1995.494215.

  3. **Comprehensive Learning Particle Swarm Optimization** *(CLPSO)*:

    J. J. Liang, A. K. Qin, P. N. Suganthan, and S. Baskar, “Comprehensive learning particle swarm optimizer for global optimization of multimodal functions,” IEEE Trans. Evol. Comput., vol. 10, no. 3, pp. 281–295, 2006, doi: 10.1109/TEVC.2005.857610.

  4. **Improved Comprehensive Learning Particle Swarm Optimization** *(ICLPSO)*:

    Z. J. Wang, Z. H. Zhan, and J. Zhang, “An improved method for comprehensive learning particle swarm optimization,” in Proceedings - 2015 IEEE Symposium Series on Computational Intelligence, SSCI 2015, 2015, pp. 218–225, doi: 10.1109/SSCI.2015.41.

### Networks

There were created such neural networks to work with :

  1. [Base Network](src/networks/base) - abstract network class (implements outputs calculation,
                                         prediction, etc.)

  2. [Base PSO Network](src/networks/base_pso) - abstract network class for PSO-like methods

  3. [Backpropagation Network](src/networks/backprop) - network for SGD with momentum

  4. [PSO Network](src/networks/pso) - network for PSO method

  5. [CLPSO Network](src/networks/clpso) - network for CLPSO method

  6. [ICLPSO Network](src/networks/iclpso) - network for ICLPSO method

Layers classes were also created for current neural networks:

  1. [Layer](src/layers/base) - abstract layer which store weights and methods
                                to initialization, outputs calculation, entries access
                                methods, etc.)

  2. [BackPropLayer](src/layers/backprop) - layer for SGD network which implements
                                            useful methods for network training

  3. [PSO_Layer](src/layers/pso) - layer for PSO-like network which implements
                                   method to calculate outputs with passed weights

For PSO-like methods neural networks have special classes called
Swarms which implement all needed functions used by each of their
methods in neural networks:

  1. [Swarm](src/swarms/base) - abstract swarm (implements base functions like
                                initialization, base variables storage, etc.)

  2. [PSO_Swarm](src/swarms/pso) - implements PSO functions

  3. [CLPSO_Swarm](src/swarms/clpso) - implements CLPSO functions

  4. [ICLPSO_Swarm](src/swarms/iclpso) - implements ICLPSO functions

### Datasets

There were chosen several datasets for classification task:

| Dataset |  Site page  | Project file |
|  :---:  |    :---:    |     :---:    |
| *Fisher's Iris*   | [Site Link](https://archive.ics.uci.edu/ml/datasets/iris) | [Project link](datasets/iris.csv) |
| *Wheat Seeds*     | [Site Link](https://archive.ics.uci.edu/ml/datasets/seeds) | [Project link](datasets/wheat-seeds.csv) |
| *Thyroid Disease* | [Site Link](https://archive.ics.uci.edu/ml/datasets/thyroid+disease) | [Project link](datasets/new-thyroid.csv) |
| *Wine*            | [Site Link](https://archive.ics.uci.edu/ml/datasets/wine) | [Project link](datasets/wine.csv) |
| *Cicada*          | [Site Link](https://www.randomservices.org/random/data/Cicada.html) | [Project link](datasets/cicada.csv) |

To use datasets for created networks, output values ​​for each dataset were replaced
with binary values ​​based on the number of unique output values (ex., if **2** unique
values (classes [1, 2], [cat, dog], ...) then final outputs must be **01** and **10**).

For this project all datasets read by method of class [Dataset](src/common/dataset)
into special variable with type 'dataset_type' (vector of pairs of inputs and outputs
vectors for used dataset) which can be easy obtained with getter function. Dataset class
also has function to split read dataset into train and test datasets for learning and
test usage. All networks accept 'dataset_type' variable as dataset for training method.
