from scipy import sparse
import numpy as np
import re

np.set_printoptions(threshold='nan')

n = 12
m = 18
Density=0.1
matrixformat='coo'
B = sparse.rand(m,n, density = Density, format = matrixformat, dtype = None)

print(B)
fo = open("sparse.txt", "w")
for i in range(0, np.size(B.col)):
    fo.write("%s %s %s\n" % (B.col[i], B.row[i], B.data[i]))
fo.close()
