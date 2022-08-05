Students: Yunyao Cheng (yunyaoc@cs.aau.dk), Kai Zhao (kaiz@cs.aau.dk)

################################################################################

Reference:

# Angus Dempster, Francois Petitjean, Geoff Webb
#
# @article{dempster_etal_2020,
#   author  = {Dempster, Angus and Petitjean, Fran\c{c}ois and Webb, Geoffrey I},
#   title   = {ROCKET: Exceptionally fast and accurate time classification using random convolutional kernels},
#   year    = {2020},
#   journal = {Data Mining and Knowledge Discovery},
#   doi     = {https://doi.org/10.1007/s10618-020-00701-z}
# }
#
# https://arxiv.org/abs/1910.13051 (preprint)

################################################################################

Reading Guide:

The file 5_rocket.pdf is the paper.
The file Assignment 5.ipynb is the runable notebook file.
The folder Car is dataset.
The file focket_functions.py is the core component of ROCKET.

################################################################################

Requirements:

torch;
Python;
Numba;
NumPy;
scikit-learn.

################################################################################

How to run Rocket:

import rocket
from sklearn.linear_model import RidgeClassifierCV

[...] # load data here. X_training, Y_training, X_test, Y_test

# transform training set and train classifier
transformer = rocket.RocketModel(input_length, 10000)
X_training_transform = transformer(X_training)
classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
classifier.fit(X_training_transform, Y_training)

# transform test set and predict
X_test_transform = apply_kernels(X_test, kernels)
predictions = classifier.predict(X_test_transform)

################################################################################
