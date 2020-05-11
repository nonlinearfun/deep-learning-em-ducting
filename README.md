# deep-learning-em-ducting
Code and results in this repository accompany the manuscript: Deep Learning Approach for Differentiating Atmospheric Ducting Within the Marine Atmospheric Boundary Layer. Hilarie Sit and Christopher J. Earls.

Dataset can be downloaded [here](https://drive.google.com/open?id=13je_sQwJzo9oEssgvuLmDJxjo8qiOsb0).

## Background
We built a two-step deep learning model to differentiate and classify evaporation ducts and surface-based ducts from sparsely sampled EM propagation data. Hyperparameters of the deep neural networks are optimized via random search on a  12-core Intel Xeon E5 microprocessor with clock speed of 2.7 GHz. Performance of individual models as well as an ensemble model is evaluated on no-noise and severely noise-contaminated test sets.

## Requirements
Python = 3.7 \
Tensorflow = 1.13.1 \
Keras = 2.3.1 \
Numpy \
Pandas

Requirements.txt is provided for recreating virtual environment

## Run hyperparameter search
Hyperparameter optimization via random search can be performed by running hypersearch.py and specifying the task:

```
python hypersearch.py --task class
```
Results from the hyperparameter search are located within the 'models' directory, in folders corresponding to the specified tasks. Models and history logs of the top five models can also be found in these folders.

## Run model evaluation
Model evaluation can be performed by running evaluation.py (specify ensemble):

```
python evaluation.py --ensemble
```
Results from evaluation are located in 'models/results'.
