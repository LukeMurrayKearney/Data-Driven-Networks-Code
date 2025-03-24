# Data-Driven Networks Code

The purpose of this repository is to assist readers in reproducing our results using the heterogeneous block model (HBM) to create age-structured contact networks as described in the following work:

Preprint page: https://www.arxiv.org/abs/2503.11527.

## Installation and Compilation

Install the package by entering

```md
git clone https://github.com/LukeMurrayKearney/Networks-from-Data-Package
```

in your terminal.

Once this is complete, enter the Networks-from-Data-Package folder and install the package requirements:

```md 
pip install -r requirements.txt
```

We now have maturin and the other dependencies installed. To compile the Rust package with pyO3 bindings in maturin enter:

```md
maturin develop --release
```

The compiled Rust code is implemented in python through the nd_python.py python file. 

The file _test_.ipynb is a jupyter notebook containing a basic use of the package to fit to data, build a network and simulate an outbreak. 

At the beginning of _test_.ipynb

```md
import nd_python as nd_p
```

imports the required functions and typing nd_p.FUNCTION(..) is used to call them. 

Feel free to use these functions in your own analysis but please remember to credit https://www.arxiv.org/abs/2503.11527 :)

## Paper Results

The accompanying python files _errors.py, _final_sellke.py, etc. were used to create the figures in the paper. 

The jupyter notebook figs.ipynb details how to recreate the figures with simulation data collected from the python files.