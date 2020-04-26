from scipy import sparse
import numpy as np
import functools

class COO:
    def __init__(self, row, col, value):
        self.row = row
        self.col = col
        self.value = value
    def __getitem__(self, key):
        if key == 0:
            return self.row
        elif key == 1:
            return self.col
        else:
            return self.value


# generate sddmm

N = 25000  # cols for sparse matrix S
M = 25000 # rows for sparse matrix S
K = 512  # the cols for D1 & D2 transposed

Density=0.001
matrixformat='coo'
eleSize = M * N * Density
B = sparse.rand(M,N, density = Density, format = matrixformat, dtype = None)

# sddmm operations
D1 = np.random.rand(M, K)
D2 = np.random.rand(K, N)
M1 = np.dot(D1, D2) # the dot product of dense matrix
M2 = B.toarray()
C = M1 * M2 # this is hadamard product of M1 and M2
Scsr = sparse.csr_matrix(C)    #to csr
Scoo = sparse.coo_matrix(C)

# note that the D1 dot with the D2 transposed 
# so actually the data form should store the D2 transposed(NxK)
fd1 = open("dense1.txt", "w")
fd2 = open("dense2.txt", "w")
for i in range(0, M):
    for j in range(0, K):
        fd1.write("%f " % (D1[i][j]))
    fd1.write("\n")
fd1.close()
D2 = np.transpose(D2)

for i in range(0, N):
    for j in range(0, K):
        fd2.write("%f " % (D2[i][j]))
    fd2.write("\n")
fd2.close()

# generate the COO form data
fo = open("sparseCOO.txt", "w")
list = []
fo.write("%d %d %d %d\n" % (M, N, K, eleSize))
for i in range(0, np.size(B.col)):
    info = COO(B.row[i], B.col[i], B.data[i])
    list.append(info)
list.sort(key=lambda s:(s[0],s[1])) 
for i in list:
    fo.write("%s %s %s\n" % (i[0],i[1],i[2]))
fo.close()

# generate the CSR form data
B = B.tocsr()
fo = open("sparseCSR.txt", "w")
fo.write("%d %d %d %d\n" % (M, N, K, eleSize))
fo.write("%d %d %d\n" % (np.size(B.indptr), np.size(B.indices), np.size(B.data)))
for i in range(0, np.size(B.indptr)):
    fo.write("%d " % B.indptr[i])
fo.write("\n")
for i in range(0, np.size(B.indices)):
    fo.write("%d " % B.indices[i])
fo.write("\n")
for i in range(0, np.size(B.data)):
    fo.write("%f " % B.data[i])
fo.write("\n")
fo.close()

f1 = open("outputCOO.txt", "w")
list = []
for i in range(0, np.size(Scoo.col)):
    info = COO(Scoo.row[i], Scoo.col[i], Scoo.data[i])
    list.append(info)
list.sort(key=lambda s:(s[0],s[1])) 
# print the right answer of sddmm
for i in list:
    f1.write("%s\n" % (i[2]))
f1.close()

f1 = open("outputCSR.txt", "w")
# print the right answer of sddmm
for i in range(0, np.size(Scsr.data)):
    f1.write("%f\n" % (Scsr.data[i]))
f1.close()

# write the CSR indptr to verify the COO->CSR C language API 
f2 = open("indptr.txt", "w")
for i in range(0, np.size(Scsr.indptr)):
    f2.write("%d " % (Scsr.indptr[i]))
f2.close()
