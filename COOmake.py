from scipy import sparse
import numpy as np
import re

n = 25
m = 30
Density=0.01
matrixformat='coo'
B = sparse.rand(m,n, density = Density, format = matrixformat, dtype = None)

print(B)
fo = open("sparse.txt", "w")
fo.write("%d %d\n" % (m, n))
for i in range(0, np.size(B.col)):
    fo.write("%s %s %s\n" % (B.col[i], B.row[i], B.data[i]))
fo.close()
