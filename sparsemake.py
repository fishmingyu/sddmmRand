from scipy import sparse
import numpy as np

# generate sddmm

N = 25  # cols for sparse matrix S
M = 40  # rows for sparse matrix S
K = 35  # the cols for D1 & D2 transposed

Density=0.01
matrixformat='coo'
B = sparse.rand(M,N, density = Density, format = matrixformat, dtype = None)

# sddmm operations
D1 = np.random.rand(M, K)
D2 = np.random.rand(K, N)
M1 = np.dot(D1, D2) # the dot product of dense matrix
M2 = B.toarray()
C = M1 * M2 # this is hadamard product of M1 and M2
S = sparse.csr_matrix(C)    #to csr

# note that the D1 dot with the D2 transposed 
# so actually the data form should store the D2 transposed(NxK)
fd1 = open("dense1.txt", "w")
fd2 = open("dense2.txt", "w")
for i in range(0, M):
    for j in range(0, K):
        fd1.write("%f " % (D1[i][j]))
    fd1.write("\n")
fd1.close()
for i in range(0, K):
    for j in range(0, N):
        fd2.write("%f " % (D2[i][j]))
    fd2.write("\n")
fd2.close()

# generate the COO form data
fo = open("sparseCOO.txt", "w")
fo.write("%d %d\n" % (M, N))
for i in range(0, np.size(B.col)):
    fo.write("%s %s %s\n" % (B.col[i], B.row[i], B.data[i]))
fo.close()

# generate the CSR form data
B = B.tocsr()
fo = open("sparseCSR.txt", "w")
f1 = open("output.txt", "w")
fo.write("%d %d %d\n" % (np.size(B.indptr), np.size(B.indices), np.size(B.data)))
for i in range(0, np.size(B.indptr)):
    fo.write("%d " % (B.indptr[i]))
fo.write("\n")
for i in range(0, np.size(B.indices)):
    fo.write("%d " % B.indices[i])
fo.write("\n")
for i in range(0, np.size(B.data)):
    fo.write("%f " % B.data[i])
fo.write("\n")
fo.close()

# print the right answer of sddmm
for i in range(0, np.size(S.data)):
    f1.write("%f\n" % (S.data[i]))
f1.close()

# write the CSR indptr to verify the COO->CSR C language API 
f2 = open("indptr.txt", "w")
for i in range(0, np.size(S.indptr)):
    f2.write("%d " % (S.indptr[i]))
f2.close()
