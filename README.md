# Predicting nuclear structure observables with Artificial Neural Networks - Student Project 2023 - Master 2 PSA
### by M. Bordeau and A. Lejeune

## Description

The need for precise nuclear mass predictions is becoming increasingly important for astrophysical applications. This contribution aim at using Artificial Neural Networks (ANNs) in order to predict the binding energy, the 2 proton and 2 neutron separation energies for nuclei with Z protons and N neutrons. The predicted data will then be compared to the experimental data from the latest Atomic Mass Evaluation (AME2020) [[1](https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt)] and to microscopic-macroscopic Duflo-Zuker model with 10 parameters (DZ10) [[2](https://arxiv.org/abs/1407.8221)].
To this end, we will solve a typical regression problem in Machine Learning (ML) by setting a feed-forward ANN using PYTHON with Keras library. 

## Installation 

In this project, DZ10 model as been provided to us by a Fortran program and small modifications has been made to this file for the sake of our use, we thus used gfortran to compile the program. 
```bash
sudo apt-get install gfortran
```

All the remaining files use PYTHON and Keras library. Keras is an open-source software library that provides a Python interface for artificial neural networks acting as an interface for the TensorFlow library.

[Check](https://www.tensorflow.org/install/source#gpu) 


## Usage

Compite the Fortran program with :
```bash
gfortran du_zu.f -o bin/duzu.exe
```

Create text file from the exe file :
```bash
./bin/duzu.exe >> raw_data/duzu.txt
```



## Roadmap



## Acknowledgement 



## License

[GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html)
