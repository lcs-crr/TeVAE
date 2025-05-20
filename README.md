# TeVAE

:exclamation: Warning :exclamation: TensorFlow 2.10 has been found to feature multiple vunerabilites, hence use this repository at your own risk. For a (for now) safe implementation of TeVAE in TensorFlow 2.15, please refer to the [PATH repository](https://github.com/lcs-crr/PATH). 

TeVAE: A Variational Autoencoder Approach for Discrete Online Anomaly Detection in Variable-state Multivariate Time-series Data

Paper corresponding to source code is submitted to the _Studies in Computational Intelligence_ journal.

Working scripts can be found in the **src** folder. 
The **data.py** script outlines the pre-processing of the data. 
The **training.py** script automates model training for all models across different seeds and data splits. 
The **evaluation.py** script outlines the steps taken to evaluate the models discussed. 

Utility scripts can be found in the **utils** folder.

Custom model classes for each of the tested approaches can be found in the **models** folder.

**requirements.txt** contains all libraries used.
