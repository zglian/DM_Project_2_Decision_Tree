from gen_dataset import *
from classifier import *

"""generate dataset"""
N = 7000 #number of data

dataset1 = gen_dataset(N)
file_path = f"inputs/dataset1-{N}.csv"
dataset1.to_csv(file_path, index = False)
print("Output dataset1")

dataset2 = gen_dataset(N)
file_path = f"inputs/dataset2-{N}.csv"
dataset2.to_csv(file_path, index = False)
print("Output dataset2")
print('==================================')
print()


"""run classifier"""
decision_tree()
naive_bayes()
svm()
knn()
print('==================================')

