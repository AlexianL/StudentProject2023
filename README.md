# Predicting nuclear structure observables with Artificial Neural Networks - Student Project 2023 - Master 2 PSA
### by M. Bordeau and A. Lejeune

## Description

The need for precise nuclear mass predictions is becoming increasingly important for astrophysical applications. This contribution aim at using Artificial Neural Networks (ANNs) in order to predict the binding energy, the 2 proton and 2 neutron separation energies for nuclei with Z protons and N neutrons. The predicted data will then be compared to the experimental data from the latest Atomic Mass Evaluation (AME2020) [[1](https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt)] and to microscopic Duflo-Zuker model with 10 parameters (DZ10) [[2](https://arxiv.org/abs/1407.8221)].
To this end, we will solve a typical regression problem in Machine Learning (ML) by setting a feed-forward ANN using PYTHON with Keras library. 

## Installation 

### PYTHON and TensorFlow

The project is mainly using PYTHON and Keras library. Keras is an open-source software library that provides a Python interface for artificial neural networks acting as an interface for the TensorFlow library. Scikit-learn is also necessary to run the codes.

TensorFlow requires specific version of some GPU drivers, it can be checked [on this address](https://www.tensorflow.org/install/source#gpu). The user needs to have a GPU compatible with the CUDA toolkit ([check if your is compatible](https://developer.nvidia.com/cuda-gpus)). If this is not the case, we have a solution using Google Colaboratory. We used TensorFlow version 2.10 which is compatible with PYTHON versions 3.7-3.10, it needs GCC version 9.3.1, build tools Bazel 5.1.1, cuDNN 8.1, CUDA 11.2. TensorFlow can be installed via conda.

For macOS, all the librairies can be easily installed with conda.



### Google Colaboratory

One of this repository author didn't have a CUDA compatible GPU. A solution has been found using Google Colaboratory which can give access to a GPU in the cloud to its users. The capacity of this GPU is far from being the best and some of our programs execution time are impacted greatly (mainly the artificial neural network training). This solution will use both Google Colaboratory and Google Drive. The user must download this repository, extract it in "My Drive" which is inside Google Drive. 


## Usage

### Duflo-Zuker with 10 parameters Fortran program

In this project, DZ10 model has been provided to us by a Fortran program and small modifications has been made to this file for the sake of our use. The output file is already in the right folder, because if you're using the Google Colaboratory solution it is not practical as you need to compile it before uploading the files on Google Drive. However if you want to see the program and compile it yourself here we show how we have done : 

We used gfortran as compiler : 
```bash
sudo apt-get install gfortran
```

We compiled it and created a text file from the .exe file :
```bash
gfortran du_zu.f -o bin/duzu.exe
./bin/duzu.exe >> raw_data/duzu.txt
```

### PYTHON notebooks

Once the repository has been uploaded to Google Drive, the user can start using the notebooks which are associated with a number to determine in which order they should be used. It is not mandatory to use the GPU offered by Google for all notebooks, but it is for the one concerning the artificial neural network. To activate this GPU, the user just has to follow what is shown on the pictures below. 

![Google Colab Tutorial 1](/images/google_colab_tutorial_1.jpg "Google Colab Tutorial 1").

The user will have to select "GPU" as the hardware accelerator. 

![Google Colab Tutorial 2](/images/google_colab_tutorial_2.jpg "Google Colab Tutorial 2").

### Complete code

In the case of execution problem with the Google Colab notebooks, it is still possible to run a version of the code which will give the same plots at the end. This code is named "6_complete_code.py".
To execute it, you must download the zip on github, then move to the directory "StudentProject2023" and use this command:

```
python3 6_complete_code.py
```
The plots will be saved into "5_plots".

## ANN error

It can happen, both for notebooks and .py code, that the learning phase of the ANN doesn't work properly. This error will not stop the execution of the code. Instead, you will see that the "loss" remains constant. If this situation happens to you, please execute again the "4_artificial_neural_netork.ipynb" or "6_complete_code.py" until this works. This error can occur on any machine.



## Roadmap



## License

[GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html)
