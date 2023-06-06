# Description

This is the code used for the final master's degree "Forecasting Betweenness Centrality with Graph Neural Networks".
It is necessary noticing that the main part of this code has been extracted from https://github.com/sunilkmaurya/GNN_Ranking since the related TFM is mainly focused on the GNN framework introduced at the referenced repository.

# Parts of the code
As it can be seen, there are some python files and folders:
* The .py files correspond to code used for the different experiments performed. Each python file is related to a set of  experiments showed in the memory of this work. In addition, the code is parametrized and easy to change in order to include variations on the experiments. The different files should be addapted to the specific needs  of each experiment.
* The data_splits folder is used for saving the different test and train splits for training and testing the different models
* The functions folder contains the definition of the PyTorch model and all the python functions used for the experiments
* The graphs folder is used for saving the differrent graphs generated
* The models folder is used in some experiments for saving some models once they are traind and avoiding training all the time the different models
* The outputs folder is used for generating some of the outputs when performing the different experiments
* The real_graphs folder contains the real graph files considered in this work (the biggest one is not available due to its size)

# Execution
For the execution of the code there is a "env.yml" file at the root folder of this repository that corresponds to the export of the conda environment used in this work. We have tried to generate, as much as possible, the environment with the same library versions used at https://github.com/sunilkmaurya/GNN_Ranking.
