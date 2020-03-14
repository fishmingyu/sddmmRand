#include "sddmmKer.h"

#define VALIDATE

//correspond with benchmark
#define checkCudaError( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    CLEANUP("Exit.");   \
    exit(EXIT_FAILURE); \
    } \
} while(0)

#define checkCuSparseError( a ) do { \
    if (CUSPARSE_STATUS_SUCCESS != (a)) { \
    fprintf(stderr, "CuSparse runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    CLEANUP("Exit.");   \
    exit(EXIT_FAILURE); \
    } \
} while (0)

#define CLEANUP(s)                      \
do {                                    \
    printf("%s", s);                  \
    if (S_csrVal) free(S_csrVal);   \
    if (S_csrRowPtr) free(S_csrRowPtr);   \
    if (S_csrColInd) free(S_csrColInd); \
    if (S_cooVal) free(S_cooVal);   \
    if (S_cooRowInd) free(S_cooRowInd);   \
    if (S_cooColInd) free(S_cooColInd); \
    if (D1_dnVal)  free(D1_dnVal);    \
    if (D2_dnVal)  free(D2_dnVal);    \
    if (O_csrVal)   free(O_csrVal);   \
    if (O_cooVal)   free(O_cooVal); \
    if (csrGoldenC) free(csrGoldenC);   \
    if (csrGoldenPy) free(csrGoldenPy);   \
    if (cooGoldenPy) free(cooGoldenPy); \
    if (dev_ScsrVal) cudaFree(dev_ScsrVal);     \
    if (dev_csrColInd) cudaFree(dev_csrColInd);         \
    if (dev_csrRowPtr) cudaFree(dev_csrRowPtr);     \
    if (dev_ScooVal) cudaFree(dev_ScooVal);     \
    if (dev_cooColInd) cudaFree(dev_cooColInd);         \
    if (dev_cooRowInd) cudaFree(dev_cooRowInd);     \
    if (dev_D1) cudaFree(dev_D1); \
    if (dev_D2) cudaFree(dev_D2); \
    if (start)      cudaEventDestroy(start);    \
    if (stop)       cudaEventDestroy(stop);     \
    cudaDeviceReset();                  \
    fflush(stdout);                     \
} while (0)

__global__ void warmup() {}

int main(int argc, char* argv[])
{
    //parameters & variables
    int S_mrows, S_ncols, D_kcols;
    unsigned long eleSize;
    //suppose D1 with the size of M x K & D2 with the size of N x K 
    //thus, D1_ncols equals to K, D2_ncols equals to , S_nrows equals to M

    //std::string coofile = "sparseCOO.txt";
    const char* csrfile = "sparseCSR.txt";
    const char* coofile = "sparseCOO.txt";
    const char* densefile1 = "dense1.txt";
    const char* densefile2 = "dense2.txt";
    const char* answerCSR = "outputCSR.txt";
    const char* answerCOO = "outputCOO.txt";

    std::vector<int> row_COO, col_IND_CSR, row_CSR, col_IND_COO;
    std::vector<data> values_CSR, values_COO;

    //Host allocate
    int* S_csrColInd = 0, * S_csrRowPtr = 0;
    int* S_cooColInd = 0, * S_cooRowInd = 0;
    data* D1_dnVal = 0, * D2_dnVal = 0;
    data* O_csrVal = 0;
    data* S_csrVal = 0;
    data* O_cooVal = 0;
    data* S_cooVal = 0;
    data* csrGoldenC = 0;
    data* cooGoldenC = 0;
    data* csrGoldenPy = 0;
    data* cooGoldenPy = 0;

    //Device allocate
    data* dev_D1 = 0, * dev_D2 = 0, * dev_Ocsr = 0;
    data* dev_ScsrVal = 0;
    data* dev_Ocoo = 0;
    data* dev_ScooVal = 0;
    
    int* dev_csrColInd = 0, * dev_csrRowPtr = 0;
    int* dev_cooColInd = 0, * dev_cooRowInd = 0;
    //time analysis
    cudaEvent_t start, stop;
    float time;
    //read from my arbitary test file
    //init SDD
    //eleSize = readCOO<float>(file, row_COO, col_IND_CSR, values_CSR, S_mrows, S_ncols);
    
    readCSR<data>(csrfile, row_CSR,
        col_IND_CSR, values_CSR, S_mrows, S_ncols, D_kcols, eleSize);
    readCOO<data>(coofile, row_COO,
        col_IND_COO, values_COO, S_mrows, S_ncols, D_kcols, eleSize);
    //read eleSize here is rundantary, only for the pursuit of neat format

    S_cooRowInd = (int*)malloc(row_COO.size() * sizeof(int));
    S_cooColInd = (int*)malloc(col_IND_COO.size() * sizeof(int));
    S_cooVal = (data*)malloc(values_COO.size() * sizeof(data));

    S_csrRowPtr = (int*)malloc(row_CSR.size() * sizeof(int));
    S_csrColInd = (int*)malloc(col_IND_CSR.size() * sizeof(int));
    S_csrVal = (data*)malloc(values_CSR.size() * sizeof(data));

    std::copy(row_CSR.begin(), row_CSR.end(), S_csrRowPtr);
    std::copy(col_IND_CSR.begin(), col_IND_CSR.end(), S_csrColInd);
    std::copy(values_CSR.begin(), values_CSR.end(), S_csrVal);

    std::copy(row_COO.begin(), row_COO.end(), S_cooRowInd);
    std::copy(col_IND_COO.begin(), col_IND_COO.end(), S_cooColInd);
    std::copy(values_COO.begin(), values_COO.end(), S_cooVal);
    //deprecated when you use the generated csr directly
    /*
   
    std::sort(row_COO.begin(), row_COO.end());
    COO_to_CSR(row_CSR, row_COO, eleSize, S_mrows);
    S_csrRowPtr = (int*)malloc(row_CSR.size() * sizeof(int));
  
     std::copy(row_CSR.begin(), row_CSR.end(), S_csrRowPtr);
    std::copy(col_IND_CSR.begin(), col_IND_CSR.end(), S_csrColInd);
    std::copy(values_CSR.begin(), values_CSR.end(), S_csrVal);
    */

    O_csrVal = (data*)malloc(eleSize * sizeof(data));
    D1_dnVal = (data*)malloc((D_kcols * S_mrows) * sizeof(data));  //D1~MxK
    D2_dnVal = (data*)malloc((D_kcols * S_ncols) * sizeof(data));   //D2~NxK
    csrGoldenC = (data*)malloc((eleSize) * sizeof(csrGoldenC[0]));
    csrGoldenPy = (data*)malloc((eleSize) * sizeof(csrGoldenPy[0]));

    O_cooVal = (data*)malloc(eleSize * sizeof(data));
    cooGoldenC = (data*)malloc((eleSize) * sizeof(csrGoldenPy[0]));
    cooGoldenPy = (data*)malloc((eleSize) * sizeof(csrGoldenPy[0]));

    if (!S_csrVal || !S_csrColInd || !S_csrRowPtr || !D1_dnVal 
        || !D2_dnVal || !O_csrVal || !csrGoldenC || !csrGoldenPy) {
        CLEANUP("Host malloc failed\n");
        return 1;
    }

    readVecMat(densefile1, D1_dnVal);
    readVecMat(densefile2, D2_dnVal);
    readVecMat(answerCSR, csrGoldenPy);
    readVecMat(answerCOO, cooGoldenPy);

    /*random assign 
    unsigned long seed = time(nullptr);
    srand(seed);
    for (int i = 0; i < D_kcols * S_mrows; i++) {
        D1_dnVal[i] = float(rand() % 10000 - 5000) / 10000;
    }
    for (int i = 0; i < D_kcols * S_ncols; i++) {
        D2_dnVal[i] = float(rand() % 10000 - 5000) / 10000;
    }
    */

    /*
#ifdef VALIDATE
    sddmmGoldenCSR<data>(S_mrows, D_kcols,
        S_csrRowPtr, S_csrColInd, D1_dnVal,
        D2_dnVal, csrGoldenC, S_csrVal);
    sddmmGoldenCOO<data>(S_mrows, D_kcols, eleSize,
        S_cooRowInd, S_cooColInd, D1_dnVal,
        D2_dnVal, cooGoldenC, S_cooVal);
#endif
    //test for basic
    for (int i = 0; i < eleSize; i++)
    {
        if (fabs(csrGoldenPy[i] - csrGoldenC[i]) > 1e-3) {
            std::cout << "WA: csrGoldenC[" << i << "] = " << csrGoldenC[i] << ", golden = " << csrGoldenPy[i] << '\n';
            break; 
        }
    }
    for (int i = 0; i < eleSize; i++)
    {
        if (fabs(cooGoldenPy[i] - cooGoldenC[i]) > 1e-3) {
            std::cout << "WA: cooGoldenC[" << i << "] = " << cooGoldenC[i] << ", golden = " << cooGoldenPy[i] << '\n';
        }
    }
    */
    // allocate device memory
    cudaDeviceReset();
    cudaSetDevice(0);
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));

    checkCudaError(cudaMalloc((void**)&dev_csrRowPtr, (S_mrows + 1) * sizeof(dev_csrRowPtr[0])));
    checkCudaError(cudaMalloc((void**)&dev_csrColInd, eleSize * sizeof(dev_csrColInd[0])));
    checkCudaError(cudaMalloc((void**)&dev_ScsrVal, eleSize * sizeof(dev_ScsrVal[0])));
    checkCudaError(cudaMalloc((void**)&dev_cooRowInd,  eleSize * sizeof(dev_cooRowInd[0])));
    checkCudaError(cudaMalloc((void**)&dev_cooColInd, eleSize * sizeof(dev_cooColInd[0])));
    checkCudaError(cudaMalloc((void**)&dev_ScsrVal, eleSize * sizeof(dev_ScsrVal[0])));
    checkCudaError(cudaMalloc((void**)&dev_ScooVal, eleSize * sizeof(dev_ScsrVal[0])));
    checkCudaError(cudaMalloc((void**)&dev_D1, S_mrows * D_kcols * sizeof(dev_D1[0])));
    checkCudaError(cudaMalloc((void**)&dev_D2, S_ncols * D_kcols * sizeof(dev_D2[0])));
    checkCudaError(cudaMalloc((void**)&dev_Ocsr, eleSize * sizeof(dev_Ocsr[0])));
    checkCudaError(cudaMalloc((void**)&dev_Ocoo, eleSize * sizeof(dev_Ocoo[0])));

    checkCudaError(cudaMemcpy(dev_csrRowPtr, S_csrRowPtr, (S_mrows + 1) * sizeof(dev_csrRowPtr[0]), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dev_csrColInd, S_csrColInd, eleSize * sizeof(dev_csrColInd[0]), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dev_ScsrVal, S_csrVal, eleSize * sizeof(dev_ScsrVal[0]), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dev_cooRowInd, S_cooRowInd, eleSize * sizeof(dev_cooRowInd[0]), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dev_cooColInd, S_cooColInd, eleSize * sizeof(dev_cooColInd[0]), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dev_ScooVal, S_cooVal, eleSize * sizeof(dev_ScooVal[0]), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dev_D1, D1_dnVal, S_mrows * D_kcols * sizeof(dev_D1[0]), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dev_D2, D2_dnVal, S_ncols * D_kcols * sizeof(dev_D2[0]), cudaMemcpyHostToDevice));

    // device warm up
    warmup <<<1, 1 >>> ();
    cudaDeviceSynchronize();
    cudaError_t cudaStat = cudaGetLastError();
    if (cudaStat != cudaSuccess)
    {
        fprintf(stderr, "launchFailedWarm-up: %s\t", cudaGetErrorString(cudaStat));
    }

#ifdef VALIDATE
    
    checkCudaError(cudaMemset((void*)dev_Ocoo, 0, eleSize * sizeof(dev_Ocoo[0])));
    sddmmWrapper<data, COOSimple, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    checkCudaError(cudaMemcpy(O_cooVal, dev_Ocoo, eleSize * sizeof(dev_Ocoo[0]), cudaMemcpyDeviceToHost));
    for (int id = 0; id < eleSize; ++id) {
        if (fabs(O_cooVal[id] - cooGoldenPy[id]) > 1e-3) {
            std::cout << "WA: O[" << id << "] = " << O_cooVal[id] << ", golden = " << cooGoldenPy[id] << '\n';
            std::cout << "the error in COOSimple" << std::endl;
            break;
        }
    }

    checkCudaError(cudaMemset((void*)dev_Ocoo, 0, eleSize * sizeof(dev_Ocoo[0])));
    sddmmWrapper<data, COOWarpShuffle, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    checkCudaError(cudaMemcpy(O_cooVal, dev_Ocoo, eleSize * sizeof(dev_Ocoo[0]), cudaMemcpyDeviceToHost));
    for (int id = 0; id < eleSize; ++id) {
        if (fabs(O_cooVal[id] - cooGoldenPy[id]) > 1e-3) {
            std::cout << "WA: O[" << id << "] = " << O_cooVal[id] << ", golden = " << cooGoldenPy[id] << '\n';
            std::cout << "the error in COOWarpShuffule" << std::endl;
            break;
        }
    }

    checkCudaError(cudaMemset((void*)dev_Ocoo, 0, eleSize * sizeof(dev_Ocoo[0])));
    sddmmWrapper<data, COOILP2, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    checkCudaError(cudaMemcpy(O_cooVal, dev_Ocoo, eleSize * sizeof(dev_Ocoo[0]), cudaMemcpyDeviceToHost));
    for (int id = 0; id < eleSize; ++id) {
        if (fabs(O_cooVal[id] - cooGoldenPy[id]) > 1e-3) {
            std::cout << "WA: O[" << id << "] = " << O_cooVal[id] << ", golden = " << cooGoldenPy[id] << '\n';
            std::cout << "the error in COOILP2" << std::endl;
            break;
        }
    }

    checkCudaError(cudaMemset((void*)dev_Ocoo, 0, eleSize * sizeof(dev_Ocoo[0])));
    sddmmWrapper<data, COOILP4, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    checkCudaError(cudaMemcpy(O_cooVal, dev_Ocoo, eleSize * sizeof(dev_Ocoo[0]), cudaMemcpyDeviceToHost));
    for (int id = 0; id < eleSize; ++id) {
        if (fabs(O_cooVal[id] - cooGoldenPy[id]) > 1e-3) {
            std::cout << "WA: O[" << id << "] = " << O_cooVal[id] << ", golden = " << cooGoldenPy[id] << '\n';
            std::cout << "the error in COOILP4" << std::endl;
            break;
        }
    }

    checkCudaError(cudaMemset((void*)dev_Ocoo, 0, eleSize * sizeof(dev_Ocoo[0])));
    sddmmWrapper<data, COOILP4ex, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    checkCudaError(cudaMemcpy(O_cooVal, dev_Ocoo, eleSize * sizeof(dev_Ocoo[0]), cudaMemcpyDeviceToHost));
    for (int id = 0; id < eleSize; ++id) {
        if (fabs(O_cooVal[id] - cooGoldenPy[id]) > 1e-3) {
            std::cout << "WA: O[" << id << "] = " << O_cooVal[id] << ", golden = " << cooGoldenPy[id] << '\n';
            std::cout << "the error in COOILP4ex" << std::endl;
            break;
        }
    }

    checkCudaError(cudaMemset((void*)dev_Ocoo, 0, eleSize * sizeof(dev_Ocoo[0])));
    sddmmWrapper<data, COOILP4Unroll, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    checkCudaError(cudaMemcpy(O_cooVal, dev_Ocoo, eleSize * sizeof(dev_Ocoo[0]), cudaMemcpyDeviceToHost));
    for (int id = 0; id < eleSize; ++id) {
        if (fabs(O_cooVal[id] - cooGoldenPy[id]) > 1e-3) {
            std::cout << "WA: O[" << id << "] = " << O_cooVal[id] << ", golden = " << cooGoldenPy[id] << '\n';
            std::cout << "the error in COOILP4Unroll" << std::endl;
            break;
        }
    }

    checkCudaError(cudaMemset((void*)dev_Ocoo, 0, eleSize * sizeof(dev_Ocoo[0])));
    sddmmWrapper<data, COOILP8, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    checkCudaError(cudaMemcpy(O_cooVal, dev_Ocoo, eleSize * sizeof(dev_Ocoo[0]), cudaMemcpyDeviceToHost));
    for (int id = 0; id < eleSize; ++id) {
        if (fabs(O_cooVal[id] - cooGoldenPy[id]) > 1e-3) {
            std::cout << "WA: O[" << id << "] = " << O_cooVal[id] << ", golden = " << cooGoldenPy[id] << '\n';
            std::cout << "the error in COOIP8" << std::endl;
            break;
        }
    }

    checkCudaError(cudaMemset((void*)dev_Ocoo, 0, eleSize * sizeof(dev_Ocoo[0])));
    sddmmWrapper<data, COOILP16, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    checkCudaError(cudaMemcpy(O_cooVal, dev_Ocoo, eleSize * sizeof(dev_Ocoo[0]), cudaMemcpyDeviceToHost));
    for (int id = 0; id < eleSize; ++id) {
        if (fabs(O_cooVal[id] - cooGoldenPy[id]) > 1e-3) {
            std::cout << "WA: O[" << id << "] = " << O_cooVal[id] << ", golden = " << cooGoldenPy[id] << '\n';
            std::cout << "the error in COOILP16" << std::endl;
            break;
        }
    }

    checkCudaError(cudaMemset((void*)dev_Ocoo, 0, eleSize * sizeof(dev_Ocoo[0])));
    sddmmWrapper<data, COOReduction, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    checkCudaError(cudaMemcpy(O_cooVal, dev_Ocoo, eleSize * sizeof(dev_Ocoo[0]), cudaMemcpyDeviceToHost));
    for (int id = 0; id < eleSize; ++id) {
        if (fabs(O_cooVal[id] - cooGoldenPy[id]) > 1e-3) {
            std::cout << "WA: O[" << id << "] = " << O_cooVal[id] << ", golden = " << cooGoldenPy[id] << '\n';
            std::cout << "the error in COOReduction" << std::endl;
            break;
        }
    }

    checkCudaError(cudaMemset((void*)dev_Ocoo, 0, eleSize * sizeof(dev_Ocoo[0])));
    sddmmWrapper<data, COOILP2Cache, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    checkCudaError(cudaMemcpy(O_cooVal, dev_Ocoo, eleSize * sizeof(dev_Ocoo[0]), cudaMemcpyDeviceToHost));
    for (int id = 0; id < eleSize; ++id) {
        if (fabs(O_cooVal[id] - cooGoldenPy[id]) > 1e-3) {
            std::cout << "WA: O[" << id << "] = " << O_cooVal[id] << ", golden = " << cooGoldenPy[id] << '\n';
            std::cout << "the error in COOILP2Cache" << std::endl;
            break;
        }
    }

    checkCudaError(cudaMemset((void*)dev_Ocoo, 0, eleSize * sizeof(dev_Ocoo[0])));
    sddmmWrapper<data, COOILP4Cache, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    checkCudaError(cudaMemcpy(O_cooVal, dev_Ocoo, eleSize * sizeof(dev_Ocoo[0]), cudaMemcpyDeviceToHost));
    for (int id = 0; id < eleSize; ++id) {
        if (fabs(O_cooVal[id] - cooGoldenPy[id]) > 1e-3) {
            std::cout << "WA: O[" << id << "] = " << O_cooVal[id] << ", golden = " << cooGoldenPy[id] << '\n';
            std::cout << "the error in COOILP4Cache" << std::endl;
            break;
        }
    }


    checkCudaError(cudaMemset((void*)dev_Ocoo, 0, eleSize * sizeof(dev_Ocoo[0])));
    sddmmWrapper<data, COOILP8Cache, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    checkCudaError(cudaMemcpy(O_cooVal, dev_Ocoo, eleSize * sizeof(dev_Ocoo[0]), cudaMemcpyDeviceToHost));
    for (int id = 0; id < eleSize; ++id) {
        if (fabs(O_cooVal[id] - cooGoldenPy[id]) > 1e-3) {
            std::cout << "WA: O[" << id << "] = " << O_cooVal[id] << ", golden = " << cooGoldenPy[id] << '\n';
            std::cout << "the error in COOILP8Cache" << std::endl;
            break;
        }
    }

    checkCudaError(cudaMemset((void*)dev_Ocsr, 0, eleSize * sizeof(dev_Ocsr[0])));
    sddmmWrapper<data, CSRSimple, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_csrRowPtr, dev_csrColInd, dev_D1, dev_D2, dev_Ocsr, dev_ScsrVal);
    checkCudaError(cudaMemcpy(O_csrVal, dev_Ocsr, eleSize * sizeof(dev_Ocsr[0]), cudaMemcpyDeviceToHost));
    for (int id = 0; id < eleSize; ++id) {
        if (fabs(O_csrVal[id] - csrGoldenPy[id]) > 1e-3) {
            std::cout << "WA: O[" << id << "] = " << O_csrVal[id] << ", golden = " << csrGoldenPy[id] << '\n';
            std::cout << "the error in CSRSimple" << std::endl;
            break;
        }
    }

    checkCudaError(cudaMemset((void*)dev_Ocsr, 0, eleSize * sizeof(dev_Ocsr[0])));
    sddmmWrapper<data, CSRCache, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_csrRowPtr, dev_csrColInd, dev_D1, dev_D2, dev_Ocsr, dev_ScsrVal);
    checkCudaError(cudaMemcpy(O_csrVal, dev_Ocsr, eleSize * sizeof(dev_Ocsr[0]), cudaMemcpyDeviceToHost));
    for (int id = 0; id < eleSize; ++id) {
        if (fabs(O_csrVal[id] - csrGoldenPy[id]) > 1e-3) {
            std::cout << "WA: O[" << id << "] = " << O_csrVal[id] << ", golden = " << csrGoldenPy[id] << '\n';
            std::cout << "the error in CSRCache" << std::endl;
            break;
        }
    }
    
    checkCudaError(cudaMemset((void*)dev_Ocsr, 0, eleSize * sizeof(dev_Ocsr[0])));
    sddmmWrapper<data, CSRWarpRec, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_csrRowPtr, dev_csrColInd, dev_D1, dev_D2, dev_Ocsr, dev_ScsrVal);
    checkCudaError(cudaMemcpy(O_csrVal, dev_Ocsr, eleSize * sizeof(dev_Ocsr[0]), cudaMemcpyDeviceToHost));
    for (int id = 0; id < eleSize; ++id) {
        if (fabs(O_csrVal[id] - csrGoldenPy[id]) > 1e-3) {
            std::cout << "WA: O[" << id << "] = " << O_csrVal[id] << ", golden = " << csrGoldenPy[id] << '\n';
            std::cout << "the error in CSRWarpRec" << std::endl;
            break;
        }
    }


    checkCudaError(cudaMemset((void*)dev_Ocsr, 0, eleSize * sizeof(dev_Ocsr[0])));
    sddmmWrapper<data, CSRWarpRec2, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_csrRowPtr, dev_csrColInd, dev_D1, dev_D2, dev_Ocsr, dev_ScsrVal);
    checkCudaError(cudaMemcpy(O_csrVal, dev_Ocsr, eleSize * sizeof(dev_Ocsr[0]), cudaMemcpyDeviceToHost));
    for (int id = 0; id < eleSize; ++id) {
        if (fabs(O_csrVal[id] - csrGoldenPy[id]) > 1e-3) {
            std::cout << "WA: O[" << id << "] = " << O_csrVal[id] << ", golden = " << csrGoldenPy[id] << '\n';
            std::cout << "the error in CSRWarpRec2" << std::endl;
            break;
        }
    }

    checkCudaError(cudaMemset((void*)dev_Ocsr, 0, eleSize * sizeof(dev_Ocsr[0])));
    sddmmWrapper<data, CSRReduction, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_csrRowPtr, dev_csrColInd, dev_D1, dev_D2, dev_Ocsr, dev_ScsrVal);
    checkCudaError(cudaMemcpy(O_csrVal, dev_Ocsr, eleSize * sizeof(dev_Ocsr[0]), cudaMemcpyDeviceToHost));
    for (int id = 0; id < eleSize; ++id) {
        if (fabs(O_csrVal[id] - csrGoldenPy[id]) > 1e-3) {
            std::cout << "WA: O[" << id << "] = " << O_csrVal[id] << ", golden = " << csrGoldenPy[id] << '\n';
            break;
        }
    }
    
    /*
    checkCudaError(cudaMemset((void*)dev_Ocsr, 0, eleSize * sizeof(dev_Ocsr[0])));
    sddmmWrapper<data, CSRCache, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_csrRowPtr, dev_csrColInd, dev_D1, dev_D2, dev_Ocsr, dev_ScsrVal);
    checkCudaError(cudaMemcpy(O_csrVal, dev_Ocsr, eleSize * sizeof(dev_Ocsr[0]), cudaMemcpyDeviceToHost));
    for (int id = 0; id < eleSize; ++id) {
        if (fabs(O_csrVal[id] - csrGoldenPy[id]) > 1e-3) {
            std::cout << "WA: O[" << id << "] = " << O_csrVal[id] << ", golden = " << csrGoldenPy[id] << '\n';
            //break;
        }
    }
    */

    checkCudaError(cudaMemset((void*)dev_Ocsr, 0, eleSize * sizeof(dev_Ocsr[0])));
    sddmmWrapper<data, CSRPar, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_csrRowPtr, dev_csrColInd, dev_D1, dev_D2, dev_Ocsr, dev_ScsrVal);
    checkCudaError(cudaMemcpy(O_csrVal, dev_Ocsr, eleSize * sizeof(dev_Ocsr[0]), cudaMemcpyDeviceToHost));
    for (int id = 0; id < eleSize; ++id) {
        if (fabs(O_csrVal[id] - csrGoldenPy[id]) > 1e-3) {
            std::cout << "WA: O[" << id << "] = " << O_csrVal[id] << ", golden = " << csrGoldenPy[id] << '\n';
            break;
        }
    }

    checkCudaError(cudaMemset((void*)dev_Ocsr, 0, eleSize * sizeof(dev_Ocsr[0])));
    sddmmWrapper<data, CSRParex, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_csrRowPtr, dev_csrColInd, dev_D1, dev_D2, dev_Ocsr, dev_ScsrVal);
    checkCudaError(cudaMemcpy(O_csrVal, dev_Ocsr, eleSize * sizeof(dev_Ocsr[0]), cudaMemcpyDeviceToHost));
    for (int id = 0; id < eleSize; ++id) {
        if (fabs(O_csrVal[id] - csrGoldenPy[id]) > 1e-3) {
            std::cout << "WA: O[" << id << "] = " << O_csrVal[id] << ", golden = " << csrGoldenPy[id] << '\n';
            break;
        }
    }

    printf("press Y to continue\n");
    if (getchar() != 'Y')
    {
        CLEANUP("\n");
        return 0;
    }

#define ITER 20

    for (int i = 0; i < 1000; i++) {
        warmup <<<1, 1 >>> ();
    }

    printf("Test data\n");


    time = 0;
    cudaEventRecord(start, 0);

    for (int i = 0; i < ITER; i++) {
        sddmmWrapper<data, COOSimple, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time /= ITER;
    printf("COOSimple ");
    printf("%.6f\n", time);

    time = 0;
    cudaEventRecord(start, 0);

    for (int i = 0; i < ITER; i++) {
        sddmmWrapper<data, COOWarpShuffle, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time /= ITER;
    printf("COOWarpShuffle ");
    printf("%.6f\n", time);

    time = 0;
    cudaEventRecord(start, 0);

    for (int i = 0; i < ITER; i++) {
        sddmmWrapper<data, COOReduction, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time /= ITER;
    printf("COORedution ");
    printf("%.6f\n", time);

    time = 0;
    cudaEventRecord(start, 0);

    for (int i = 0; i < ITER; i++) {
        sddmmWrapper<data, COOILP2, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time /= ITER;
    printf("COOILP2 ");
    printf("%.6f\n", time);

    time = 0;
    cudaEventRecord(start, 0);

    for (int i = 0; i < ITER; i++) {
        sddmmWrapper<data, COOILP2Cache, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time /= ITER;
    printf("COOILP2Cache ");
    printf("%.6f\n", time);

    time = 0;
    cudaEventRecord(start, 0);

    for (int i = 0; i < ITER; i++) {
        sddmmWrapper<data, COOILP4Cache, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time /= ITER;
    printf("COOILP4Cache ");
    printf("%.6f\n", time);


    time = 0;
    cudaEventRecord(start, 0);

    for (int i = 0; i < ITER; i++) {
        sddmmWrapper<data, COOILP8Cache, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time /= ITER;
    printf("COOILP8Cache ");
    printf("%.6f\n", time);

    time = 0;
    cudaEventRecord(start, 0);

    for (int i = 0; i < ITER; i++) {
        sddmmWrapper<data, COOILP4, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time /= ITER;
    printf("COOILP4 ");
    printf("%.6f\n", time);

    time = 0;
    cudaEventRecord(start, 0);

    for (int i = 0; i < ITER; i++) {
        sddmmWrapper<data, COOILP4ex, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time /= ITER;
    printf("COOILP4ex ");
    printf("%.6f\n", time);

    time = 0;
    cudaEventRecord(start, 0);

    for (int i = 0; i < ITER; i++) {
        sddmmWrapper<data, COOILP4Unroll, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time /= ITER;
    printf("COOILP4Unroll ");
    printf("%.6f\n", time);

    time = 0;
    cudaEventRecord(start, 0);

    for (int i = 0; i < ITER; i++) {
        sddmmWrapper<data, COOILP8, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time /= ITER;
    printf("COOILP8 ");
    printf("%.6f\n", time);

    time = 0;
    cudaEventRecord(start, 0);

    for (int i = 0; i < ITER; i++) {
        sddmmWrapper<data, COOILP16, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_cooRowInd, dev_cooColInd, dev_D1, dev_D2, dev_Ocoo, dev_ScooVal);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time /= ITER;
    printf("COOILP16 ");
    printf("%.6f\n", time);

    time = 0;
    cudaEventRecord(start, 0);
    checkCudaError(cudaMemset((void*)dev_Ocsr, 0, eleSize * sizeof(dev_Ocsr[0])));
    for (int i = 0; i < ITER; i++) {
        sddmmWrapper<data, CSRPar, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_csrRowPtr, dev_csrColInd, dev_D1, dev_D2, dev_Ocsr, dev_ScsrVal);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time /= ITER;
    printf("CSRPar ");
    printf("%.6f\n", time);

    time = 0;
    cudaEventRecord(start, 0);
    checkCudaError(cudaMemset((void*)dev_Ocsr, 0, eleSize * sizeof(dev_Ocsr[0])));
    for (int i = 0; i < ITER; i++) {
        sddmmWrapper<data, CSRParex, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_csrRowPtr, dev_csrColInd, dev_D1, dev_D2, dev_Ocsr, dev_ScsrVal);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time /= ITER;
    printf("CSRParex ");
    printf("%.6f\n", time);

    time = 0;
    cudaEventRecord(start, 0);
    checkCudaError(cudaMemset((void*)dev_Ocsr, 0, eleSize * sizeof(dev_Ocsr[0])));
    for (int i = 0; i < ITER; i++) {
        sddmmWrapper<data, CSRWarpRec, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_csrRowPtr, dev_csrColInd, dev_D1, dev_D2, dev_Ocsr, dev_ScsrVal);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time /= ITER;
    printf("CSRWarpReduc ");
    printf("%.6f\n", time);

    time = 0;
    cudaEventRecord(start, 0);
    checkCudaError(cudaMemset((void*)dev_Ocsr, 0, eleSize * sizeof(dev_Ocsr[0])));
    for (int i = 0; i < ITER; i++) {
        sddmmWrapper<data, CSRWarpRec2, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_csrRowPtr, dev_csrColInd, dev_D1, dev_D2, dev_Ocsr, dev_ScsrVal);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time /= ITER;
    printf("CSRWarpReduc2 ");
    printf("%.6f\n", time);

    time = 0;
    cudaEventRecord(start, 0);
    checkCudaError(cudaMemset((void*)dev_Ocsr, 0, eleSize * sizeof(dev_Ocsr[0])));
    for (int i = 0; i < ITER; i++) {
        sddmmWrapper<data, CSRCache, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_csrRowPtr, dev_csrColInd, dev_D1, dev_D2, dev_Ocsr, dev_ScsrVal);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time /= ITER;
    printf("CSRShareMemory ");
    printf("%.6f\n", time);

    time = 0;
    cudaEventRecord(start, 0);
    checkCudaError(cudaMemset((void*)dev_Ocsr, 0, eleSize * sizeof(dev_Ocsr[0])));
    for (int i = 0; i < ITER; i++) {
        sddmmWrapper<data, CSRReduction, BLOCKDIMZ>(S_mrows, D_kcols, eleSize, dev_csrRowPtr, dev_csrColInd, dev_D1, dev_D2, dev_Ocsr, dev_ScsrVal);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time /= ITER;
    printf("CSRReduction ");
    printf("%.6f\n", time);

    CLEANUP("\n");

    return 0;
#endif //VALIDATE
}
