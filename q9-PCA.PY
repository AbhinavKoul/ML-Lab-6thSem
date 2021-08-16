import pandas as pd
import numpy as np

X = np.randint(10,50,100).reshape(20,5)

X_meaned = X - np.mean(X,axis = 0)

cov_mat = np.cov(X_meaned,rowvar = False)

eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

sorted_index = np.argsort(eigen_values)[::-1]
sorted_eigen_vectors = eigen_vectors[:,sorted_index]

n_comp = 2
eigen_vectors_subset = sorted_eigen_vectors[:,0:n_comp]

X_reduced = np.dot(eigen_vectors_subset.transpose(),X.transpose()).transpose()