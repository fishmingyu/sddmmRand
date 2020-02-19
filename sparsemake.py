from scipy import sparse
import numpy as np

# generate sddmm

n = 25  # cols for sparse matrix S
m = 40  # rows for sparse matrix S
k = 35  # the cols for D1 & D2 transposed

Density=0.01
matrixformat='coo'
B = sparse.rand(m,n, density = Density, format = matrixformat, dtype = None)

# sddmm operations
D1 = np.random.rand(m, k)
D2 = np.random.rand(k, n)
M1 = np.dot(D1, D2) # the dot product of dense matrix
M2 = B.toarray()
C = M1 * M2 # this is hadamard product of M1 and M2
S = sparse.csr_matrix(C)    #to csr

# generate the COO form data
fo = open("sparseCOO.txt", "w")
fo.write("%d %d\n" % (m, n))
for i in range(0, np.size(B.col)):
    fo.write("%s %s %s\n" % (B.col[i], B.row[i], B.data[i]))
fo.close()

# generate the CSR form data
B = B.tocsr()
fo = open("sparseCSR.txt", "w")
f1 = open("output.txt", "w")
fo.write("%d %d\n" % (m, n))
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
