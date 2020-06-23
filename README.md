GPyOpt_DFT
===========

This is an extended version GPyOpt package [GPyOpt homepage](http://sheffieldml.github.io/GPyOpt/). A new acquisition function utilizing DFT data, EI_DFT, has been added to the package. This acqusition function is designed to be used for optimizing ternary MAPbI/FAPbI/CsPbI perovskite mixtures. The description of the original GPyOpt repository follows.

Gaussian process optimization using [GPy](http://sheffieldml.github.io/GPy/). Performs global optimization with different acquisition functions. Among other functionalities, it is possible to use GPyOpt to optimize physical experiments (sequentially or in batches) and tune the parameters of Machine Learning algorithms. It is able to handle large data sets via sparse Gaussian process models.

* [GPyOpt homepage](http://sheffieldml.github.io/GPyOpt/)
* [Tutorial Notebooks](http://nbviewer.ipython.org/github/SheffieldML/GPyOpt/blob/master/manual/index.ipynb)
* [Online documentation](http://gpyopt.readthedocs.io/)

[![licence](https://img.shields.io/badge/licence-BSD-blue.svg)](http://opensource.org/licenses/BSD-3-Clause)  [![develstat](https://travis-ci.org/SheffieldML/GPyOpt.svg?branch=master)](https://travis-ci.org/SheffieldML/GPyOpt) [![covdevel](http://codecov.io/github/SheffieldML/GPyOpt/coverage.svg?branch=master)](http://codecov.io/github/SheffieldML/GPyOpt?branch=master) [![Research software impact](http://depsy.org/api/package/pypi/GPyOpt/badge.svg)](http://depsy.org/package/python/GPyOpt)


### Citation

    @Misc{gpyopt2016,
      author =   {The GPyOpt authors},
      title =    {{GPyOpt}: A Bayesian Optimization framework in python},
      howpublished = {\url{http://github.com/SheffieldML/GPyOpt}},
      year = {2016}
    }

Getting started
===============

Clone the repository in GitHub and add it to your $PYTHONPATH.

```bash
    git clone https://github.com/srags/GPyOpt_DFT.git
    cd GPyOpt_DFT
    python setup.py install
```

Dependencies:
------------------------
  - GPy
  - paramz
  - numpy
  - scipy
  - matplotlib
  - DIRECT (optional)
  - cma (optional)
  - pyDOE (optional)
  - sobol_seq (optional)

You can install dependencies by running:
```
pip install -r requirements.txt
```








