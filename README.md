# Predicting nuclear structure observables with Artificial Neural Networks - Student Project 2023 - Master 2 PSA
### by M. Bordeau and A. Lejeune

## Description

The need for precise nuclear mass predictions is becoming increasingly important for astrophysical applications. This contribution aim at using Artificial Neural Networks (ANNs) in order to predict the binding energy, the 2 proton and 2 neutron separation energies for nuclei with Z protons and N neutrons. The predicted data will then be compared to the experimental data from the latest Atomic Mass Evaluation (AME2020) [[1](https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt)] and to microscopic Duflo-Zuker model with 10 parameters (DZ10) [[2](https://arxiv.org/abs/1407.8221)].
To this end, we will solve a typical regression problem in Machine Learning (ML) by setting a feed-forward ANN using PYTHON with Keras library. 

## Installation 

### Fortran compiler

In this project, DZ10 model as been provided to us by a Fortran program and small modifications has been made to this file for the sake of our use, we thus used gfortran to compile the program. 
```bash
sudo apt-get install gfortran
```

### PYTHON and TensorFlow

All the remaining files use PYTHON and Keras library. Keras is an open-source software library that provides a Python interface for artificial neural networks acting as an interface for the TensorFlow library.

TensorFlow requires specific version of some GPU drivers, it can be checked [on this address](https://www.tensorflow.org/install/source#gpu). The user needs to have a GPU compatible with the CUDA toolkit ([check if your is compatible](https://developer.nvidia.com/cuda-gpus)). If this is not the case, we have a solution using Google Colaboratory. We used TensorFlow version 2.10 which is compatible with PYTHON versions 3.7-3.10, it needs GCC version 9.3.1, build tools Bazel 5.1.1, cuDNN 8.1, CUDA 11.2. 

### Google Colaboratory

One of this repository author didn't have a CUDA compatible GPU. A solution has been found using Google Colaboratory which can give access to a GPU in the cloud to its users. The capacity of this GPU is far from being the best and some of our programs execution time are impacted greatly (mainly the artificial neural network training). This solution will use both Google Colaboratory and Google Drive. The user must download this repository, extract it in "My Drive" which is inside Google Drive. 

![alt text for screen readers](/images/google_colab_tutorial_1.jpg "Text to show on mouseover").

![alt text for screen readers](/images/to/google_colab_tutorial_2.jpg "Text to show on mouseover").

## Usage

Compile the Fortran program with :
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
