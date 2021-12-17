# comod
[![Github release](https://img.shields.io/github/release/dih5/comod.svg)](https://github.com/dih5/comod/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/comod.svg)](https://pypi.python.org/pypi/comod)

[![license MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/Dih5/comod/master/LICENSE.txt)

[![Documentation Status](https://readthedocs.org/projects/comod/badge/?version=latest)](http://comod.readthedocs.io/en/latest/?badge=latest)

Compartmental modelling Python package

## Preview
![alt tag](https://raw.github.com/dih5/comod/master/imgs/motivation1.png)
![alt tag](https://raw.github.com/dih5/comod/master/imgs/motivation2.png)

Check the [docs](https://comod.readthedocs.io/en/latest/) to see more.


## Features
- Define a model with simple rules as strings or with custom functions.
- Pre-defined models like SIR, SIS, SEIR, ...
- Community-extensions of models.
- Solve numerically for fixed or time-dependent parameters.
- Best-fit to existing data, posibly using time windows.
- Create compartment graphs.
- Export LaTeX.
- Export to Wolfram Language (Mathematica).


## Installation
Assuming you have a [Python3](https://www.python.org/) distribution with [pip](https://pip.pypa.io/en/stable/installing/), the latest pypi release can be installed with:
```
pip3 install comod
```
To install the optional dependencies you can run
```
pip3 install 'comod[extras]'
```
Mind the quotes.


## Developer information
### Instalation

To install a development version, cd to the directory with this file and:

```
pip3 install -e '.[test]'
```
As an alternative, a virtualenv might be used to install the package:
```
# Prepare a clean virtualenv and activate it
virtualenv -p /usr/bin/python3.6 venv
source venv/bin/activate
# Install the package
pip3 install -e '.[test]'
```

### Documentation
To generate the documentation, the *docs* extra dependencies must be installed.

To generate an html documentation with sphinx run:
```
make docs
```

To generate a PDF documentation using LaTeX:
```
make pdf
```



### Test
To run the unitary tests:
```
make test
```
