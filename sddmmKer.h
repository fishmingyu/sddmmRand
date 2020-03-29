#pragma once
#ifndef sddmmKer_H
#define sddmmKer_H

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <ctime>

#define QUOCEIL(x, y) (((x)+(y)-1) / (y))

#define MIN(a, b) (a?b:a<b)
#define MAX(a, b) (a?b:a>b)
#define D2ROW 500   //the rows of the D2
#define BLOCKDIMZ 1
const int threadsPerBlock = 32;
const int blocksPerGrid = MIN(32, ((D2ROW + threadsPerBlock - 1)/threadsPerBlock));
typedef float data;

template<typename T>
void sddmmGoldenCSR(int S_mrows, int D_kcols,
    int* S_csrRowPtr, int* S_csrColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_csrVal, T* S_csrVal)
    //D1 dot with D2 transposed
    //suppose D1 with the size of M x K & D2 with the size of N x K 
{
    for (int i = 0; i < S_mrows; i++)
    {
        int lb = S_csrRowPtr[i];
        int hb = S_csrRowPtr[i + 1];
        int offset1, offset2;
        T acc = 0;
        for (int ptr = lb; ptr < hb; ptr++)
        {
            offset1 = i * D_kcols;
            offset2 = S_csrColInd[ptr] * D_kcols;
            for (int k = 0; k < D_kcols; k++)
            {
                acc += D1_dnVal[k + offset1] * D2_dnVal[k + offset2];
            }
            O_csrVal[ptr] = acc * S_csrVal[ptr];
            acc = 0;
        }
    }
}

template<typename T>
void sddmmGoldenCOO(int S_mrows, int D_kcols, unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
    //D1 dot with D2 transposed
    //suppose D1 with the size of M x K & D2 with the size of N x K 
{
    for (int i = 0; i < Size; i++)
    {
        int row = S_cooRowInd[i];
        int col = S_cooColInd[i];
        int offset1, offset2;
        T acc = 0;
        offset1 = row * D_kcols;
        offset2 = col * D_kcols;
        for (int k = 0; k < D_kcols; k++)
        {
            acc += D1_dnVal[k + offset1] * D2_dnVal[k + offset2];
        }
        O_cooVal[i] = acc;
    }
    for (int i = 0; i < Size; i++)
    {
        O_cooVal[i] *= S_cooVal[i];
    }
}

template<typename T>
__global__ void sddmmCSRSimple(int S_mrows, int D_kcols,
    int* S_csrRowPtr, int* S_csrColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_csrVal, T* S_csrVal)  //Simple loop chucking
//S_mrows:rows number 
{
    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx < S_mrows)
    {
        int lb = S_csrRowPtr[tidx];
        int hb = S_csrRowPtr[tidx + 1];
        int offset1, offset2;
        T acc = 0;
        for (int ptr = lb; ptr < hb; ptr++)
        {
            offset1 = tidx * D_kcols;
            offset2 = S_csrColInd[ptr] * D_kcols;
            for (int k = 0; k < D_kcols; k++) //need to be optimized
            {
                //the k^th 
                acc += D1_dnVal[k + offset1] * D2_dnVal[k + offset2];  
            }
            O_csrVal[ptr] = acc * S_csrVal[ptr];
            acc = 0;
        }
    }
}

//the kernel for COO directly
template<typename T>
__global__ void sddmmCOOCacheCast(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    __shared__ T temp[32];
    int eid = blockIdx.x;
    int cid = threadIdx.x + threadIdx.y * blockDim.x; //the id for column
    temp[cid % 32] = 0;
    __syncthreads();
    int offset1 = S_cooRowInd[eid] * D_kcols + cid;
    int offset2 = S_cooColInd[eid] * D_kcols + cid ;
    __syncthreads();
    T multi = D1_dnVal[offset1] * D2_dnVal[offset2] * S_cooVal[eid];
    atomicAdd(&temp[cid % 32], multi);
    __syncthreads();
    if(threadIdx.x < 32)
    {
        multi = temp[threadIdx.x];
        for(int stride = 16; stride > 0; stride >>= 1)
            multi += __shfl_down_sync(0xffffffff, multi, stride);
    }
    if(threadIdx.x == 0)
    {
        O_cooVal[eid] = multi;
    }
}

//float mode use intrinsics
__global__ void sddmmCOOCacheCastSFU(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, float* D1_dnVal,
    float* D2_dnVal, float* O_cooVal, float* S_cooVal)
{
    __shared__ float temp[32];
    int eid = blockIdx.x;
    int cid = threadIdx.x + threadIdx.y * blockDim.x; //the id for column
    temp[cid % 32] = 0;
    __syncthreads();
    int offset1 = S_cooRowInd[eid] * D_kcols + cid;
    int offset2 = S_cooColInd[eid] * D_kcols + cid ;
    __syncthreads();
    float multi =  __fmul_rn(__fmul_rn(D1_dnVal[offset1], D2_dnVal[offset2]), S_cooVal[eid]);
    atomicAdd(&temp[cid % 32], multi);
    __syncthreads();
    if(threadIdx.x < 32)
    {
        multi = temp[threadIdx.x];
        for(int stride = 16; stride > 0; stride >>= 1)
            multi += __shfl_down_sync(0xffffffff, multi, stride);
    }
    if(threadIdx.x == 0)
    {
        O_cooVal[eid] = multi;
    }
}

//the kernel for COO shuffle
template<typename T>
__global__ void sddmmCOOShuffleKernel(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    const int eid = blockIdx.x << 2;
    const int cid = threadIdx.x + threadIdx.y * blockDim.x; //the id for column
    int offset1[4], offset2[4];
    T multi[4] = {0, 0, 0 ,0};
    if (cid < D_kcols) {
        if (eid != Size - 3){
#pragma simd
            offset1[0] = S_cooRowInd[eid] * D_kcols;
            offset1[1] = S_cooRowInd[eid + 1] * D_kcols;
            offset1[2] = S_cooRowInd[eid + 2] * D_kcols;
            offset1[3] = S_cooRowInd[eid + 3] * D_kcols;
#pragma simd
            offset2[0] = S_cooColInd[eid] * D_kcols;
            offset2[1] = S_cooColInd[eid + 1] * D_kcols;
            offset2[2] = S_cooColInd[eid + 2] * D_kcols;
            offset2[3] = S_cooColInd[eid + 3] * D_kcols;
#pragma simd
            multi[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
            multi[1] = D1_dnVal[offset1[1] + cid] * D2_dnVal[offset2[1] + cid] * S_cooVal[eid + 1];
            multi[2] = D1_dnVal[offset1[2] + cid] * D2_dnVal[offset2[2] + cid] * S_cooVal[eid + 2];
            multi[3] = D1_dnVal[offset1[3] + cid] * D2_dnVal[offset2[3] + cid] * S_cooVal[eid + 3];
        }
        else{
#pragma unroll 4
            for(int i = 0;i < Size % 4;i++){
                offset1[i] = S_cooRowInd[eid + i] * D_kcols;
                offset2[i] = S_cooColInd[eid + i] * D_kcols;
                multi[i] = D1_dnVal[offset1[i] + cid] * D2_dnVal[offset2[i] + cid] * S_cooVal[eid + i];
            }
        }
    }
    for (int stride = 16; stride > 0; stride >>= 1)
    {
        multi[0] += __shfl_down_sync(0xffffffff, multi[0], stride);
        multi[1] += __shfl_down_sync(0xffffffff, multi[1], stride);
        multi[2] += __shfl_down_sync(0xffffffff, multi[2], stride);
        multi[3] += __shfl_down_sync(0xffffffff, multi[3], stride);
    }
    if (threadIdx.x == 0) {
        if(eid != Size - 1)
        {
#pragma simd
            atomicAdd(&O_cooVal[eid], multi[0]);
            atomicAdd(&O_cooVal[eid + 1], multi[1]);
            atomicAdd(&O_cooVal[eid + 2], multi[2]);
            atomicAdd(&O_cooVal[eid + 3], multi[3]);
        }
        else
        {
            for(int i = 0;i < Size % 4;i++)
                atomicAdd(&O_cooVal[eid + i], multi[i]);
        }
    }
}

template<typename T>
__global__ void sddmmCOOReductionSimpleKernel(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    extern __shared__ T multi[];
    const int eid = blockIdx.x;
    const int cid = (threadIdx.y << 9) + threadIdx.x;
    int offset1, offset2;
    offset1 = S_cooRowInd[eid] * D_kcols;
    offset2 = S_cooColInd[eid] * D_kcols;
    multi[cid] = 0;
    if (cid < D_kcols)
        multi[cid] = D1_dnVal[offset1 + cid] * D2_dnVal[offset2 + cid] * S_cooVal[eid];
    __syncthreads();
    if (threadIdx.x < 256)
    {
        multi[cid] += multi[cid + 256];
    }
    __syncthreads();
    if (threadIdx.x < 128)
    {
        multi[cid] += multi[cid + 128];
    }
    __syncthreads();
    if (threadIdx.x < 64)
    {
        multi[cid] += multi[cid + 64];
    }
    __syncthreads();
    if (threadIdx.x < 32)
    {
        multi[cid] += multi[cid + 32];
    }
    __syncthreads();
    if (threadIdx.x < 16)
    {
        multi[cid] += multi[cid + 16];
        __syncwarp();
        multi[cid] += multi[cid + 8];
        __syncwarp();
        multi[cid] += multi[cid + 4];
        __syncwarp();
        multi[cid] += multi[cid + 2];
    }
    if (threadIdx.x == 0) {
        atomicAdd(&O_cooVal[eid], multi[threadIdx.y << 9] + multi[(threadIdx.y << 9) + 1]);
    }
}


//the kernel for COO reduction with register plus shmem optimized
template<typename T>
__global__ void sddmmCOOReductionKernel(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    extern __shared__ T multi[];
    const int eid = blockIdx.x;
    const int cid = (threadIdx.y << 9) + threadIdx.x;
    int offset1, offset2;
    offset1 = S_cooRowInd[eid] * D_kcols;
    offset2 = S_cooColInd[eid] * D_kcols;
    T* const smem_ptr = &multi[cid];
    T localResult = 0;
    if (cid < D_kcols)
        localResult = D1_dnVal[offset1 + cid] * D2_dnVal[offset2 + cid] * S_cooVal[eid];
    *smem_ptr = localResult;
    __syncthreads();
    if (threadIdx.x < 256)
    {
        localResult += *(smem_ptr + 256);
        *smem_ptr = localResult;
    }
    __syncthreads();
    if (threadIdx.x < 128)
    {
        localResult += *(smem_ptr + 128);
        *smem_ptr = localResult;
    }
    __syncthreads();
    if (threadIdx.x < 64)
    {
        localResult += *(smem_ptr + 64);
        *smem_ptr = localResult;
    }
    __syncthreads();
    if (threadIdx.x < 32)
    {
        localResult += *(smem_ptr + 32);
        *smem_ptr = localResult;
    }
    __syncthreads();
    if (threadIdx.x < 16)
    {
        localResult += *(smem_ptr + 16);
        *smem_ptr = localResult;
        localResult += *(smem_ptr + 8);
        *smem_ptr = localResult;
        localResult += *(smem_ptr + 4);
        *smem_ptr = localResult;
        localResult += *(smem_ptr + 2);
        *smem_ptr = localResult;
    }
    if (threadIdx.x == 0) {
        localResult += *(smem_ptr + 1);
        atomicAdd(&O_cooVal[eid], localResult);
    }
}

//actually reduce the amount of blocks 
template<typename T>
__global__ void sddmmCOOILP2ShReduc(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    extern __shared__ T multi[];
    const int multiOffset = QUOCEIL(D_kcols, 512) * 512;
    const int eid = (blockIdx.x) << 1;
    const int cid = (threadIdx.y << 9) + threadIdx.x;
    int offset1[2], offset2[2];
    T localResult[2] = { 0, 0 };
    //Compute the vector dot production using ILP2
    //Be careful with the logic judgement
    if (cid < D_kcols){
        if (eid != Size - 1){
            offset1[0] = S_cooRowInd[eid] * D_kcols;
            offset1[1] = S_cooRowInd[eid + 1] * D_kcols;
            offset2[0] = S_cooColInd[eid] * D_kcols;
            offset2[1] = S_cooColInd[eid + 1] * D_kcols;
            localResult[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
            localResult[1] = D1_dnVal[offset1[1] + cid] * D2_dnVal[offset2[1] + cid] * S_cooVal[eid + 1];
        }
        else{
            offset1[0] = S_cooRowInd[eid] * D_kcols;
            offset2[0] = S_cooColInd[eid] * D_kcols;
            localResult[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
        }
    }

    T* const smem_ptr[2] = { &multi[cid], &multi[cid + multiOffset]};
    *smem_ptr[0] = localResult[0];
    *smem_ptr[1] = localResult[1];

    __syncthreads();
    if (threadIdx.x < 256){
        localResult[0] += *(smem_ptr[0] + 256);
        localResult[1] += *(smem_ptr[1] + 256);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
    }

    __syncthreads();
    if (threadIdx.x < 128){
        localResult[0] += *(smem_ptr[0] + 128);
        localResult[1] += *(smem_ptr[1] + 128);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
    }

    __syncthreads();
    if (threadIdx.x < 64){
        localResult[0] += *(smem_ptr[0] + 64);
        localResult[1] += *(smem_ptr[1] + 64);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
    }

    __syncthreads();
    if (threadIdx.x < 32){
        localResult[0] += *(smem_ptr[0] + 32);
        localResult[1] += *(smem_ptr[1] + 32);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
    }

    __syncthreads();
    if (threadIdx.x < 16){
        localResult[0] += *(smem_ptr[0] + 16);
        localResult[1] += *(smem_ptr[1] + 16);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
        localResult[0] += *(smem_ptr[0] + 8);
        localResult[1] += *(smem_ptr[1] + 8);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
        localResult[0] += *(smem_ptr[0] + 4);
        localResult[1] += *(smem_ptr[1] + 4);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
        localResult[0] += *(smem_ptr[0] + 2);
        localResult[1] += *(smem_ptr[1] + 2);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
    }
    if (threadIdx.x == 0) {
        if(eid != Size - 1)
        {
            atomicAdd(&O_cooVal[eid], multi[(threadIdx.y << 9)] + multi[(threadIdx.y << 9) + 1]);
            atomicAdd(&O_cooVal[eid + 1], multi[multiOffset + (threadIdx.y << 9)] + multi[multiOffset + (threadIdx.y << 9) + 1]);
        }
        else
        {
            atomicAdd(&O_cooVal[eid], multi[(threadIdx.y << 9)] + multi[(threadIdx.y << 9) + 1]);
        }
    }
}

template<typename T>
__global__ void sddmmCOOILP2ShReducEX(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    extern __shared__ T multi[];
    const int eid = (blockIdx.x) << 1;
    const int cid = (threadIdx.y << 9) + threadIdx.x;
    int offset1[2], offset2[2];
    T localResult[2] = { 0, 0 };
    //Compute the vector dot production using ILP2
    //Be careful with the logic judgement
    if (cid < D_kcols){
        if (eid != Size - 1){
            offset1[0] = S_cooRowInd[eid] * D_kcols;
            offset1[1] = S_cooRowInd[eid + 1] * D_kcols;
            offset2[0] = S_cooColInd[eid] * D_kcols;
            offset2[1] = S_cooColInd[eid + 1] * D_kcols;
            localResult[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
            localResult[1] = D1_dnVal[offset1[1] + cid] * D2_dnVal[offset2[1] + cid] * S_cooVal[eid + 1];
        }
        else{
            offset1[0] = S_cooRowInd[eid] * D_kcols;
            offset2[0] = S_cooColInd[eid] * D_kcols;
            localResult[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
        }
    }

    T* const smem_ptr[2] = { &multi[cid << 1], &multi[(cid << 1) + 1]};
    *smem_ptr[0] = localResult[0];
    *smem_ptr[1] = localResult[1];

    __syncthreads();
    if (threadIdx.x < 256){
        localResult[0] += *(smem_ptr[0] + 512);
        localResult[1] += *(smem_ptr[1] + 512);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
    }

    __syncthreads();
    if (threadIdx.x < 128){
        localResult[0] += *(smem_ptr[0] + 256);
        localResult[1] += *(smem_ptr[1] + 256);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
    }

    __syncthreads();
    if (threadIdx.x < 64){
        localResult[0] += *(smem_ptr[0] + 128);
        localResult[1] += *(smem_ptr[1] + 128);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
    }

    __syncthreads();
    if (threadIdx.x < 32){
        localResult[0] += *(smem_ptr[0] + 64);
        localResult[1] += *(smem_ptr[1] + 64);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
    }

    __syncwarp();
    if (threadIdx.x < 16){
        localResult[0] += *(smem_ptr[0] + 32);
        localResult[1] += *(smem_ptr[1] + 32);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
    }
    __syncwarp();
    if(threadIdx.x < 8)
    {
        localResult[0] += *(smem_ptr[0] + 16);
        localResult[1] += *(smem_ptr[1] + 16);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
    }
    __syncwarp();
    if(threadIdx.x < 4)
    {
        localResult[0] += *(smem_ptr[0] + 8);
        localResult[1] += *(smem_ptr[1] + 8);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
    }
    __syncwarp();
    if(threadIdx.x < 2)
    {
        localResult[0] += *(smem_ptr[0] + 4);
        localResult[1] += *(smem_ptr[1] + 4);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
    }
    if (threadIdx.x == 0) {
        if(eid != Size - 1)
        {
            localResult[0] += *(smem_ptr[0] + 2);
            localResult[1] += *(smem_ptr[1] + 2);
            atomicAdd(&O_cooVal[eid], localResult[0]);
            atomicAdd(&O_cooVal[eid + 1], localResult[1]);
        }
        else
        {
            localResult[0] += *(smem_ptr[0] + 2);
            atomicAdd(&O_cooVal[eid], localResult[0]);
        }
    }
}

template<typename T>
__global__ void sddmmCOOILP4ShReduc(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    extern __shared__ T multi[];
    int eid = (blockIdx.x) << 2;
    int cid = (threadIdx.y << 9) + threadIdx.x;
    int offset1[4], offset2[4];
    T localResult[4] = { 0, 0, 0, 0 };
    //Compute the vector dot production using ILP4
    //Be careful with the logic judgement
    if (cid < D_kcols){
        if (eid != Size - 3){
            offset1[0] = S_cooRowInd[eid] * D_kcols;
            offset1[1] = S_cooRowInd[eid + 1] * D_kcols;
            offset1[2] = S_cooRowInd[eid + 2] * D_kcols;
            offset1[3] = S_cooRowInd[eid + 3] * D_kcols;
            offset2[0] = S_cooColInd[eid] * D_kcols;
            offset2[1] = S_cooColInd[eid + 1] * D_kcols;
            offset2[2] = S_cooColInd[eid + 2] * D_kcols;
            offset2[3] = S_cooColInd[eid + 3] * D_kcols;
            localResult[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
            localResult[1] = D1_dnVal[offset1[1] + cid] * D2_dnVal[offset2[1] + cid] * S_cooVal[eid + 1];
            localResult[2] = D1_dnVal[offset1[2] + cid] * D2_dnVal[offset2[2] + cid] * S_cooVal[eid + 2];
            localResult[3] = D1_dnVal[offset1[3] + cid] * D2_dnVal[offset2[3] + cid] * S_cooVal[eid + 3];
        }
        else{
            for(int i = 0;i < Size % 4; i++){
                offset1[i] = S_cooRowInd[eid + i] * D_kcols;
                offset2[i] = S_cooColInd[eid + i] * D_kcols;
                localResult[i] = D1_dnVal[offset1[i] + cid] * D2_dnVal[offset2[i] + cid] * S_cooVal[eid + i];
            }
        }
    }

    T* const smem_ptr[4] = { &multi[cid << 2], &multi[(cid << 2) + 1], &multi[(cid << 2) + 2], &multi[(cid << 2) + 3] };
    *smem_ptr[0] = localResult[0];
    *smem_ptr[1] = localResult[1];
    *smem_ptr[2] = localResult[2];
    *smem_ptr[3] = localResult[3];

    __syncthreads();
    if (threadIdx.x < 256){
        localResult[0] += *(smem_ptr[0] + 1024);
        localResult[1] += *(smem_ptr[1] + 1024);
        localResult[2] += *(smem_ptr[2] + 1024);
        localResult[3] += *(smem_ptr[3] + 1024);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
        *smem_ptr[2] = localResult[2];
        *smem_ptr[3] = localResult[3];
    }

    __syncthreads();
    if (threadIdx.x < 128){
        localResult[0] += *(smem_ptr[0] + 512);
        localResult[1] += *(smem_ptr[1] + 512);
        localResult[2] += *(smem_ptr[2] + 512);
        localResult[3] += *(smem_ptr[3] + 512);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
        *smem_ptr[2] = localResult[2];
        *smem_ptr[3] = localResult[3];
    }

    __syncthreads();
    if (threadIdx.x < 64){
        localResult[0] += *(smem_ptr[0] + 256);
        localResult[1] += *(smem_ptr[1] + 256);
        localResult[2] += *(smem_ptr[2] + 256);
        localResult[3] += *(smem_ptr[3] + 256);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
        *smem_ptr[2] = localResult[2];
        *smem_ptr[3] = localResult[3];
    }

    __syncthreads();
    if (threadIdx.x < 32){
        localResult[0] += *(smem_ptr[0] + 128);
        localResult[1] += *(smem_ptr[1] + 128);
        localResult[2] += *(smem_ptr[2] + 128);
        localResult[3] += *(smem_ptr[3] + 128);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
        *smem_ptr[2] = localResult[2];
        *smem_ptr[3] = localResult[3];
    }

    __syncwarp();
    if (threadIdx.x < 16){
        localResult[0] += *(smem_ptr[0] + 64);
        localResult[1] += *(smem_ptr[1] + 64);
        localResult[2] += *(smem_ptr[2] + 64);
        localResult[3] += *(smem_ptr[3] + 64);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
        *smem_ptr[2] = localResult[2];
        *smem_ptr[3] = localResult[3];
    }
    __syncwarp();
    if(threadIdx.x < 8)
    {
        localResult[0] += *(smem_ptr[0] + 32);
        localResult[1] += *(smem_ptr[1] + 32);
        localResult[2] += *(smem_ptr[2] + 32);
        localResult[3] += *(smem_ptr[3] + 32);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
        *smem_ptr[2] = localResult[2];
        *smem_ptr[3] = localResult[3];
    }
    __syncwarp();
    if(threadIdx.x < 4)
    {
        localResult[0] += *(smem_ptr[0] + 16);
        localResult[1] += *(smem_ptr[1] + 16);
        localResult[2] += *(smem_ptr[2] + 16);
        localResult[3] += *(smem_ptr[3] + 16);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
        *smem_ptr[2] = localResult[2];
        *smem_ptr[3] = localResult[3];
    }
    __syncwarp();
    if(threadIdx.x < 2)
    {
        localResult[0] += *(smem_ptr[0] + 8);
        localResult[1] += *(smem_ptr[1] + 8);
        localResult[2] += *(smem_ptr[2] + 8);
        localResult[3] += *(smem_ptr[3] + 8);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
        *smem_ptr[2] = localResult[2];
        *smem_ptr[3] = localResult[3];
    }
    if (threadIdx.x == 0) {
        if(eid != Size - 3)
        {
            localResult[0] += *(smem_ptr[0] + 4);
            localResult[1] += *(smem_ptr[1] + 4);
            localResult[2] += *(smem_ptr[2] + 4);
            localResult[3] += *(smem_ptr[3] + 4);
            
            atomicAdd(&O_cooVal[eid], localResult[0]);
            atomicAdd(&O_cooVal[eid + 1], localResult[1]);
            atomicAdd(&O_cooVal[eid + 2], localResult[2]);
            atomicAdd(&O_cooVal[eid + 3], localResult[3]);
        }
        else
        {
            for(int i = 0;i < Size % 4; i++){
                localResult[i] += *(smem_ptr[i] + 4);
                atomicAdd(&O_cooVal[eid + i], localResult[i]);
            }
        }
    }
}


template<typename T>
__global__ void sddmmCOOILP4LoopShReduc(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    extern __shared__ T multi[];
    int eid = (blockIdx.x) << 2;
    int cid = (threadIdx.y << 9) + threadIdx.x;
    int offset1[4], offset2[4];
    T localResult[4] = { 0, 0, 0, 0 };
    //Compute the vector dot production using ILP4
    //Be careful with the logic judgement
    if (cid < D_kcols){
        if (eid != Size - 3){
            offset1[0] = S_cooRowInd[eid] * D_kcols;
            offset1[1] = S_cooRowInd[eid + 1] * D_kcols;
            offset1[2] = S_cooRowInd[eid + 2] * D_kcols;
            offset1[3] = S_cooRowInd[eid + 3] * D_kcols;
            offset2[0] = S_cooColInd[eid] * D_kcols;
            offset2[1] = S_cooColInd[eid + 1] * D_kcols;
            offset2[2] = S_cooColInd[eid + 2] * D_kcols;
            offset2[3] = S_cooColInd[eid + 3] * D_kcols;
            localResult[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
            localResult[1] = D1_dnVal[offset1[1] + cid] * D2_dnVal[offset2[1] + cid] * S_cooVal[eid + 1];
            localResult[2] = D1_dnVal[offset1[2] + cid] * D2_dnVal[offset2[2] + cid] * S_cooVal[eid + 2];
            localResult[3] = D1_dnVal[offset1[3] + cid] * D2_dnVal[offset2[3] + cid] * S_cooVal[eid + 3];
        }
        else{
            for(int i = 0;i < Size % 4; i++){
                offset1[i] = S_cooRowInd[eid + i] * D_kcols;
                offset2[i] = S_cooColInd[eid + i] * D_kcols;
                localResult[i] = D1_dnVal[offset1[i] + cid] * D2_dnVal[offset2[i] + cid] * S_cooVal[eid + i];
            }
        }
    }

    T* const smem_ptr[4] = { &multi[cid << 2], &multi[(cid << 2) + 1], &multi[(cid << 2) + 2], &multi[(cid << 2) + 3] };
    *smem_ptr[0] = localResult[0];
    *smem_ptr[1] = localResult[1];
    *smem_ptr[2] = localResult[2];
    *smem_ptr[3] = localResult[3];
#pragma unroll 4
    for(int i = (blockDim.x << 1);i > 64;i >>= 1)
    {
        __syncthreads();
        if(threadIdx.x < (i >> 2)){
            localResult[0] += *(smem_ptr[0] + i);
            localResult[1] += *(smem_ptr[1] + i);
            localResult[2] += *(smem_ptr[2] + i);
            localResult[3] += *(smem_ptr[3] + i);
            *smem_ptr[0] = localResult[0];
            *smem_ptr[1] = localResult[1];
            *smem_ptr[2] = localResult[2];
            *smem_ptr[3] = localResult[3];
        }
    }
#pragma unroll 4
    for(int i = 64;i > 4;i >>= 1)
    {
        __syncwarp();
        if(threadIdx.x < (i >> 2)){
            localResult[0] += *(smem_ptr[0] + i);
            localResult[1] += *(smem_ptr[1] + i);
            localResult[2] += *(smem_ptr[2] + i);
            localResult[3] += *(smem_ptr[3] + i);
            *smem_ptr[0] = localResult[0];
            *smem_ptr[1] = localResult[1];
            *smem_ptr[2] = localResult[2];
            *smem_ptr[3] = localResult[3];
        }
    }   

    if (threadIdx.x == 0) {
        if(eid != Size - 3)
        {
#pragma simd
            localResult[0] += *(smem_ptr[0] + 4);
            localResult[1] += *(smem_ptr[1] + 4);
            localResult[2] += *(smem_ptr[2] + 4);
            localResult[3] += *(smem_ptr[3] + 4);
            
            atomicAdd(&O_cooVal[eid], localResult[0]);
            atomicAdd(&O_cooVal[eid + 1], localResult[1]);
            atomicAdd(&O_cooVal[eid + 2], localResult[2]);
            atomicAdd(&O_cooVal[eid + 3], localResult[3]);
        }
        else
        {
            for(int i = 0;i < Size % 4; i++){
                localResult[i] += *(smem_ptr[i] + 2);
                atomicAdd(&O_cooVal[eid + i], localResult[i]);
            }
        }
    }
}

template<typename T>
__global__ void sddmmCOOILP8ShReduc(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    extern __shared__ T multi[];
    int multiOffset = QUOCEIL(D_kcols, 512) * 512;
    int eid = (blockIdx.x) << 3;
    int cid = (threadIdx.y << 9) + threadIdx.x;
    int offset1[8], offset2[8];
    T localResult[8] = { 0, 0, 0, 0 ,0 ,0, 0, 0};
    //Compute the vector dot production using ILP8
    //Be careful with the logic judgement
    if (cid < D_kcols){
        if (eid != Size - 7){
            offset1[0] = S_cooRowInd[eid] * D_kcols;
            offset1[1] = S_cooRowInd[eid + 1] * D_kcols;
            offset1[2] = S_cooRowInd[eid + 2] * D_kcols;
            offset1[3] = S_cooRowInd[eid + 3] * D_kcols;
            offset1[4] = S_cooRowInd[eid + 4] * D_kcols;
            offset1[5] = S_cooRowInd[eid + 5] * D_kcols;
            offset1[6] = S_cooRowInd[eid + 6] * D_kcols;
            offset1[7] = S_cooRowInd[eid + 7] * D_kcols;
            offset2[0] = S_cooColInd[eid] * D_kcols;
            offset2[1] = S_cooColInd[eid + 1] * D_kcols;
            offset2[2] = S_cooColInd[eid + 2] * D_kcols;
            offset2[3] = S_cooColInd[eid + 3] * D_kcols;
            offset2[4] = S_cooColInd[eid + 4] * D_kcols;
            offset2[5] = S_cooColInd[eid + 5] * D_kcols;
            offset2[6] = S_cooColInd[eid + 6] * D_kcols;
            offset2[7] = S_cooColInd[eid + 7] * D_kcols;
            localResult[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
            localResult[1] = D1_dnVal[offset1[1] + cid] * D2_dnVal[offset2[1] + cid] * S_cooVal[eid + 1];
            localResult[2] = D1_dnVal[offset1[2] + cid] * D2_dnVal[offset2[2] + cid] * S_cooVal[eid + 2];
            localResult[3] = D1_dnVal[offset1[3] + cid] * D2_dnVal[offset2[3] + cid] * S_cooVal[eid + 3];
            localResult[4] = D1_dnVal[offset1[4] + cid] * D2_dnVal[offset2[4] + cid] * S_cooVal[eid + 4];
            localResult[5] = D1_dnVal[offset1[5] + cid] * D2_dnVal[offset2[5] + cid] * S_cooVal[eid + 5];
            localResult[6] = D1_dnVal[offset1[6] + cid] * D2_dnVal[offset2[6] + cid] * S_cooVal[eid + 6];
            localResult[7] = D1_dnVal[offset1[7] + cid] * D2_dnVal[offset2[7] + cid] * S_cooVal[eid + 7];
        }
        else{
            for(int i = 0;i < Size % 8; i++){
                offset1[i] = S_cooRowInd[eid + i] * D_kcols;
                offset2[i] = S_cooColInd[eid + i] * D_kcols;
                localResult[i] = D1_dnVal[offset1[i] + cid] * D2_dnVal[offset2[i] + cid] * S_cooVal[eid + i];
            }
        }
    }

    T* const smem_ptr[8] = { 
        &multi[cid], &multi[cid + multiOffset], 
        &multi[cid + 2 * multiOffset], &multi[cid + 3 * multiOffset] ,
        &multi[cid + 4 * multiOffset], &multi[cid + 5 * multiOffset] ,
        &multi[cid + 6 * multiOffset], &multi[cid + 7 * multiOffset] };
    *smem_ptr[0] = localResult[0];
    *smem_ptr[1] = localResult[1];
    *smem_ptr[2] = localResult[2];
    *smem_ptr[3] = localResult[3];
    *smem_ptr[4] = localResult[4];
    *smem_ptr[5] = localResult[5];
    *smem_ptr[6] = localResult[6];
    *smem_ptr[7] = localResult[7];

    for(int i = blockDim.x >> 1;i > 16;i >>= 1)
    {
        __syncthreads();
        if(threadIdx.x < i){
            localResult[0] += *(smem_ptr[0] + i);
            localResult[1] += *(smem_ptr[1] + i);
            localResult[2] += *(smem_ptr[2] + i);
            localResult[3] += *(smem_ptr[3] + i);
            localResult[4] += *(smem_ptr[4] + i);
            localResult[5] += *(smem_ptr[5] + i);
            localResult[6] += *(smem_ptr[6] + i);
            localResult[7] += *(smem_ptr[7] + i);
            *smem_ptr[0] = localResult[0];
            *smem_ptr[1] = localResult[1];
            *smem_ptr[2] = localResult[2];
            *smem_ptr[3] = localResult[3];
            *smem_ptr[4] = localResult[4];
            *smem_ptr[5] = localResult[5];
            *smem_ptr[6] = localResult[6];
            *smem_ptr[7] = localResult[7];
        }
    }
    for(int i = 16;i > 1;i >>= 1)
    {
        __syncwarp();
        if(threadIdx.x < i){
            localResult[0] += *(smem_ptr[0] + i);
            localResult[1] += *(smem_ptr[1] + i);
            localResult[2] += *(smem_ptr[2] + i);
            localResult[3] += *(smem_ptr[3] + i);
            localResult[4] += *(smem_ptr[4] + i);
            localResult[5] += *(smem_ptr[5] + i);
            localResult[6] += *(smem_ptr[6] + i);
            localResult[7] += *(smem_ptr[7] + i);
            *smem_ptr[0] = localResult[0];
            *smem_ptr[1] = localResult[1];
            *smem_ptr[2] = localResult[2];
            *smem_ptr[3] = localResult[3];
            *smem_ptr[4] = localResult[4];
            *smem_ptr[5] = localResult[5];
            *smem_ptr[6] = localResult[6];
            *smem_ptr[7] = localResult[7];
        }
    }

    if (threadIdx.x == 0) {
        if(eid != Size - 7)
        {
            localResult[0] += *(smem_ptr[0] + 1);
            localResult[1] += *(smem_ptr[1] + 1);
            localResult[2] += *(smem_ptr[2] + 1);
            localResult[3] += *(smem_ptr[3] + 1);
            localResult[4] += *(smem_ptr[4] + 1);
            localResult[5] += *(smem_ptr[5] + 1);
            localResult[6] += *(smem_ptr[6] + 1);
            localResult[7] += *(smem_ptr[7] + 1);
            atomicAdd(&O_cooVal[eid], localResult[0]);
            atomicAdd(&O_cooVal[eid + 1], localResult[1]);
            atomicAdd(&O_cooVal[eid + 2], localResult[2]);
            atomicAdd(&O_cooVal[eid + 3], localResult[3]);
            atomicAdd(&O_cooVal[eid + 4], localResult[4]);
            atomicAdd(&O_cooVal[eid + 5], localResult[5]);
            atomicAdd(&O_cooVal[eid + 6], localResult[6]);
            atomicAdd(&O_cooVal[eid + 7], localResult[7]);
        }
        else
        {
            for(int i = 0;i < Size % 8; i++){
                localResult[i] += *(smem_ptr[i] + 1);
                atomicAdd(&O_cooVal[eid + i], localResult[i]);
            }
        }
    }
}

template<typename T>
__global__ void sddmmCOOILP2CacheCast(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    __shared__ T temp[64];
    const int eid = (blockIdx.x + blockIdx.y * gridDim.x) << 1;
    const int cid = threadIdx.x + threadIdx.y * blockDim.x;
    temp[cid % 64] = 0;
    __syncthreads();
    int offset1[2], offset2[2];
    T shflD[2];
    T multi[2] = {0, 0};
    if (eid != Size - 1)
    {
        offset1[0] = S_cooRowInd[eid] * D_kcols;
        offset1[1] = S_cooRowInd[eid + 1] * D_kcols;
        offset2[0] = S_cooColInd[eid] * D_kcols;
        offset2[1] = S_cooColInd[eid + 1] * D_kcols;
        __syncthreads();
        multi[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
        multi[1] = D1_dnVal[offset1[1] + cid] * D2_dnVal[offset2[1] + cid] * S_cooVal[eid + 1];
        atomicAdd(&temp[cid % 32], multi[0]);
        atomicAdd(&temp[cid % 32 + 32], multi[1]);
        __syncthreads();
        if(threadIdx.x < 32)
        {
            shflD[0] = temp[threadIdx.x];
            shflD[1] = temp[threadIdx.x + 32];
            for (int stride = 16; stride > 0; stride >>= 1)
            {
                shflD[0] += __shfl_down_sync(0xffffffff, shflD[0], stride, 32);
                shflD[1] += __shfl_down_sync(0xffffffff, shflD[1], stride, 32);
            }
        }
        if(threadIdx.x == 0)
        {
            atomicAdd(&O_cooVal[eid], shflD[0]);
            atomicAdd(&O_cooVal[eid + 1], shflD[1]);
        }
    }
    else
    {
        offset1[0] = S_cooRowInd[eid] * D_kcols;
        offset2[0] = S_cooColInd[eid] * D_kcols;
        multi[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
        atomicAdd(&O_cooVal[eid], multi[0]);
    }
}

template<typename T>
__global__ void sddmmCOOILP4CacheCast(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    __shared__ T temp[128];
    const int eid = (blockIdx.x) << 2;
    const int cid = threadIdx.x;
    int offset1[4], offset2[4];
    temp[cid % 128] = 0;
    __syncthreads();
    T multi[4] = { 0, 0, 0, 0 };
    if (eid < Size - 3)
    {
        int4 tempRow = *(reinterpret_cast<int4 *> (S_cooRowInd + eid));
        offset1[0] = tempRow.x * D_kcols;
        offset1[1] = tempRow.y * D_kcols;
        offset1[2] = tempRow.z * D_kcols;
        offset1[3] = tempRow.w * D_kcols;
        int4 tempCol = *(reinterpret_cast<int4 *> (S_cooColInd + eid));
        offset2[0] = tempCol.x * D_kcols;
        offset2[1] = tempCol.y * D_kcols;
        offset2[2] = tempCol.z * D_kcols;
        offset2[3] = tempCol.w * D_kcols;
        __syncthreads();
        multi[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
        multi[1] = D1_dnVal[offset1[1] + cid] * D2_dnVal[offset2[1] + cid] * S_cooVal[eid + 1];
        multi[2] = D1_dnVal[offset1[2] + cid] * D2_dnVal[offset2[2] + cid] * S_cooVal[eid + 2];
        multi[3] = D1_dnVal[offset1[3] + cid] * D2_dnVal[offset2[3] + cid] * S_cooVal[eid + 3];
        atomicAdd(&temp[cid % 32], multi[0]);
        atomicAdd(&temp[cid % 32 + 32], multi[1]);
        atomicAdd(&temp[cid % 32 + 64], multi[2]);
        atomicAdd(&temp[cid % 32 + 96], multi[3]);
        __syncthreads();
        if(threadIdx.x < 32)
        {
            multi[0] = temp[threadIdx.x];
            multi[1] = temp[threadIdx.x + 32];
            multi[2] = temp[threadIdx.x + 64];
            multi[3] = temp[threadIdx.x + 96];
            for (int stride = 16; stride > 0; stride >>= 1)
            {
                multi[0] += __shfl_down_sync(0xffffffff, multi[0], stride, 32);
                multi[1] += __shfl_down_sync(0xffffffff, multi[1], stride, 32);
                multi[2] += __shfl_down_sync(0xffffffff, multi[2], stride, 32);
                multi[3] += __shfl_down_sync(0xffffffff, multi[3], stride, 32);
            }
        }
        if(threadIdx.x == 0)
        {
            *(reinterpret_cast<float4*>(O_cooVal + eid)) = *(reinterpret_cast<float4*>(multi));
        }
    }
    else
    {
        switch (Size % 4)
        {
        case 1:
        {
            offset1[0] = S_cooRowInd[eid] * D_kcols;
            offset2[0] = S_cooColInd[eid] * D_kcols;
            multi[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
            atomicAdd(&O_cooVal[eid], multi[0]);
            break;
        }
        case 2:
        {
            offset1[0] = S_cooRowInd[eid] * D_kcols;
            offset1[1] = S_cooRowInd[eid + 1] * D_kcols;
            offset2[0] = S_cooColInd[eid] * D_kcols;
            offset2[1] = S_cooColInd[eid + 1] * D_kcols;
            multi[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
            multi[1] = D1_dnVal[offset1[1] + cid] * D2_dnVal[offset2[1] + cid] * S_cooVal[eid + 1];
            atomicAdd(&O_cooVal[eid], multi[0]);
            atomicAdd(&O_cooVal[eid + 1], multi[1]);
            break;
        }
        case 3:
        {
            offset1[0] = S_cooRowInd[eid] * D_kcols;
            offset1[1] = S_cooRowInd[eid + 1] * D_kcols;
            offset1[2] = S_cooRowInd[eid + 2] * D_kcols;
            offset2[0] = S_cooColInd[eid] * D_kcols;
            offset2[1] = S_cooColInd[eid + 1] * D_kcols;
            offset2[2] = S_cooColInd[eid + 2] * D_kcols;
            multi[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
            multi[1] = D1_dnVal[offset1[1] + cid] * D2_dnVal[offset2[1] + cid] * S_cooVal[eid + 1];
            multi[2] = D1_dnVal[offset1[2] + cid] * D2_dnVal[offset2[2] + cid] * S_cooVal[eid + 2];
            atomicAdd(&O_cooVal[eid], multi[0]);
            atomicAdd(&O_cooVal[eid + 1], multi[1]);
            atomicAdd(&O_cooVal[eid + 2], multi[2]);
            break;
        }
        default:
            break;
        }
    }
}

template<typename T>
__global__ void sddmmCOOILP4CacheCastNext(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    __shared__ T temp[128];
    const int eid = (blockIdx.x) << 2;
    const int cid = (threadIdx.x << 1);
    int offset1[4], offset2[4];
    temp[threadIdx.x % 128] = 0;
    __syncthreads();
    T multi[4] = { 0, 0, 0, 0 };
    if (eid < Size - 3)
    {
        int4 tempRow = *(reinterpret_cast<int4 *> (S_cooRowInd + eid));
        offset1[0] = tempRow.x * D_kcols + cid;
        offset1[1] = tempRow.y * D_kcols + cid;
        offset1[2] = tempRow.z * D_kcols + cid;
        offset1[3] = tempRow.w * D_kcols + cid;
        int4 tempCol = *(reinterpret_cast<int4 *> (S_cooColInd + eid));
        offset2[0] = tempCol.x * D_kcols + cid;
        offset2[1] = tempCol.y * D_kcols + cid;
        offset2[2] = tempCol.z * D_kcols + cid;
        offset2[3] = tempCol.w * D_kcols + cid;
        __syncthreads();
        if(cid < D_kcols - 1)
        {
            multi[0] = D1_dnVal[offset1[0]] * D2_dnVal[offset2[0]] + D1_dnVal[offset1[0] + 1] * D2_dnVal[offset2[0] + 1];
            multi[0] *= S_cooVal[eid];
            multi[1] = D1_dnVal[offset1[1]] * D2_dnVal[offset2[1]] + D1_dnVal[offset1[1] + 1] * D2_dnVal[offset2[1] + 1];
            multi[1] *= S_cooVal[eid + 1];
            multi[2] = D1_dnVal[offset1[2]] * D2_dnVal[offset2[2]] + D1_dnVal[offset1[2] + 1] * D2_dnVal[offset2[2] + 1];
            multi[2] *= S_cooVal[eid + 2];
            multi[3] = D1_dnVal[offset1[3]] * D2_dnVal[offset2[3]] + D1_dnVal[offset1[3] + 1] * D2_dnVal[offset2[3] + 1];
            multi[3] *= S_cooVal[eid + 3];
        }
        else
        {
            multi[0] = D1_dnVal[offset1[0]] * D2_dnVal[offset2[0]] * S_cooVal[eid];
            multi[1] = D1_dnVal[offset1[1]] * D2_dnVal[offset2[1]] * S_cooVal[eid + 1];
            multi[2] = D1_dnVal[offset1[2]] * D2_dnVal[offset2[2]] * S_cooVal[eid + 2];
            multi[3] = D1_dnVal[offset1[3]] * D2_dnVal[offset2[3]] * S_cooVal[eid + 3]; 
        }

        atomicAdd(&temp[threadIdx.x % 32], multi[0]);
        atomicAdd(&temp[threadIdx.x % 32 + 32], multi[1]);
        atomicAdd(&temp[threadIdx.x % 32 + 64], multi[2]);
        atomicAdd(&temp[threadIdx.x % 32 + 96], multi[3]);
        __syncthreads();

        if(threadIdx.x < 32)
        {
            T* const tempp = temp + threadIdx.x; 
            multi[0] = *tempp;
            multi[1] = *(tempp + 32);
            multi[2] = *(tempp + 64);
            multi[3] = *(tempp + 96);
            for (int stride = 16; stride > 0; stride >>= 1)
            {
                multi[0] += __shfl_down_sync(0xffffffff, multi[0], stride, 32);
                multi[1] += __shfl_down_sync(0xffffffff, multi[1], stride, 32);
                multi[2] += __shfl_down_sync(0xffffffff, multi[2], stride, 32);
                multi[3] += __shfl_down_sync(0xffffffff, multi[3], stride, 32);
            }
        }
        if(threadIdx.x == 0)
        {
            *(reinterpret_cast<float4*>(O_cooVal + eid)) = *(reinterpret_cast<float4*>(multi));
        }
    }
    else
    {
        switch (Size % 4)
        {
        case 1:
        {
            offset1[0] = S_cooRowInd[eid] * D_kcols + cid;
            offset2[0] = S_cooColInd[eid] * D_kcols + cid;
            __syncthreads();
            if(cid < D_kcols - 1)
            {
                multi[0] = D1_dnVal[offset1[0]] * D2_dnVal[offset2[0]]
                + D1_dnVal[offset1[0] + 1] * D2_dnVal[offset2[0] + 1];
                multi[0] *= S_cooVal[eid];
            }
            else
            {
                multi[0] = D1_dnVal[offset1[0]] * D2_dnVal[offset2[0]];
                multi[0] *= S_cooVal[eid];
            }
            atomicAdd(&temp[threadIdx.x % 32], multi[0]);
            __syncthreads();
            if(threadIdx.x < 32)
            {
                T* const tempp = temp + threadIdx.x; 
                multi[0] = *tempp;
                for (int stride = 16; stride > 0; stride >>= 1)
                {
                    multi[0] += __shfl_down_sync(0xffffffff, multi[0], stride, 32);
                }
            }
            if(threadIdx.x == 0)             
                O_cooVal[eid] = multi[0];
            break;
        }
        case 2:
        {
            int2 tempRow = *(reinterpret_cast<int2 *> (S_cooRowInd + eid));
            int2 tempCol = *(reinterpret_cast<int2 *> (S_cooColInd + eid));
            offset1[0] = tempRow.x * D_kcols + cid;
            offset1[1] = tempRow.y * D_kcols + cid;
            offset2[0] = tempCol.x * D_kcols + cid;
            offset2[1] = tempCol.y * D_kcols + cid;
            __syncthreads();
            if(cid < D_kcols - 1)
            {
                multi[0] = D1_dnVal[offset1[0]] * D2_dnVal[offset2[0]]
                + D1_dnVal[offset1[0] + 1] * D2_dnVal[offset2[0] + 1];
                multi[1] = D1_dnVal[offset1[1]] * D2_dnVal[offset2[1]]
                + D1_dnVal[offset1[1] + 1] * D2_dnVal[offset2[1] + 1];
                multi[0] *= S_cooVal[eid];
                multi[1] *= S_cooVal[eid + 1];
            }
            else
            {
                multi[0] = D1_dnVal[offset1[0]] * D2_dnVal[offset2[0]];
                multi[1] = D1_dnVal[offset1[1]] * D2_dnVal[offset2[1]];
                multi[0] *= S_cooVal[eid];
                multi[1] *= S_cooVal[eid + 1];
            }
            atomicAdd(&temp[threadIdx.x % 32], multi[0]);
            atomicAdd(&temp[threadIdx.x % 32 + 32], multi[1]);
            __syncthreads();
            if(threadIdx.x < 32)
            {
                T* const tempp = temp + threadIdx.x; 
                multi[0] = *tempp;
                multi[1] = *(tempp + 32);
                for (int stride = 16; stride > 0; stride >>= 1)
                {
                    multi[0] += __shfl_down_sync(0xffffffff, multi[0], stride, 32);
                    multi[1] += __shfl_down_sync(0xffffffff, multi[1], stride, 32);
                }
            }
            if(threadIdx.x == 0)  
            {
                *(reinterpret_cast<float2*>(O_cooVal + eid)) = *(reinterpret_cast<float2*>(multi));
            }
            break;
        }
        case 3:
        {
            int3 tempRow = *(reinterpret_cast<int3 *> (S_cooRowInd + eid));
            int3 tempCol = *(reinterpret_cast<int3 *> (S_cooColInd + eid));
            offset1[0] = tempRow.x * D_kcols + cid;
            offset1[1] = tempRow.y * D_kcols + cid;
            offset1[2] = tempRow.z * D_kcols + cid;
            offset2[0] = tempCol.x * D_kcols + cid;
            offset2[1] = tempCol.y * D_kcols + cid;
            offset2[2] = tempCol.z * D_kcols + cid;
            __syncthreads();
            if(cid < D_kcols - 1)
            {
                multi[0] = D1_dnVal[offset1[0]] * D2_dnVal[offset2[0]]
                + D1_dnVal[offset1[0] + 1] * D2_dnVal[offset2[0] + 1];
                multi[1] = D1_dnVal[offset1[1]] * D2_dnVal[offset2[1]] 
                + D1_dnVal[offset1[1] + 1] * D2_dnVal[offset2[1] + 1];
                multi[2] = D1_dnVal[offset1[2]] * D2_dnVal[offset2[2]]
                + D1_dnVal[offset1[2] + 1] * D2_dnVal[offset2[2] + 1]; 
                multi[0] *= S_cooVal[eid];
                multi[1] *= S_cooVal[eid + 1];
                multi[2] *= S_cooVal[eid + 2];
            }
            else
            {
                multi[0] = D1_dnVal[offset1[0]] * D2_dnVal[offset2[0]] * S_cooVal[eid];
                multi[1] = D1_dnVal[offset1[1]] * D2_dnVal[offset2[1]] * S_cooVal[eid + 1];
                multi[2] = D1_dnVal[offset1[2]] * D2_dnVal[offset2[2]] * S_cooVal[eid + 2];
            }
            atomicAdd(&temp[threadIdx.x % 32], multi[0]);
            atomicAdd(&temp[threadIdx.x % 32 + 32], multi[1]);
            atomicAdd(&temp[threadIdx.x % 32 + 64], multi[2]);
            __syncthreads();
            if(threadIdx.x < 32)
            {
                T* const tempp = temp + threadIdx.x; 
                multi[0] = *tempp;
                multi[1] = *(tempp + 32);
                multi[2] = *(tempp + 64);
                for (int stride = 16; stride > 0; stride >>= 1)
                {
                    multi[0] += __shfl_down_sync(0xffffffff, multi[0], stride, 32);
                    multi[1] += __shfl_down_sync(0xffffffff, multi[1], stride, 32);
                    multi[2] += __shfl_down_sync(0xffffffff, multi[2], stride, 32);
                }
            }
            if(threadIdx.x == 0)  
            {           
                *(reinterpret_cast<float3*>(O_cooVal + eid)) = *(reinterpret_cast<float3*>(multi));
            }
            break;
        }
        default:
            break;
        }
    }
}

template<typename T>
__global__ void sddmmCOOILP4CacheCastNext4(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    __shared__ T temp[128];
    const int eid = (blockIdx.x) << 2;
    const int cid = (threadIdx.x << 2);
    int offset1[4], offset2[4];
    temp[threadIdx.x % 128] = 0;
    __syncthreads();
    int offsetCid = threadIdx.x % 32;
    T* const tempoffset = temp + offsetCid;
    T multi[4] = { 0, 0, 0, 0 };
    if (eid < Size - 3)
    {
        int4 tempRow = *(reinterpret_cast<int4 *> (S_cooRowInd + eid));
        offset1[0] = tempRow.x * D_kcols + cid;
        offset1[1] = tempRow.y * D_kcols + cid;
        offset1[2] = tempRow.z * D_kcols + cid;
        offset1[3] = tempRow.w * D_kcols + cid;
        int4 tempCol = *(reinterpret_cast<int4 *> (S_cooColInd + eid));
        offset2[0] = tempCol.x * D_kcols + cid;
        offset2[1] = tempCol.y * D_kcols + cid;
        offset2[2] = tempCol.z * D_kcols + cid;
        offset2[3] = tempCol.w * D_kcols + cid;
        __syncthreads();
        if(cid < D_kcols - 3)
        {
            multi[0] = D1_dnVal[offset1[0]] * D2_dnVal[offset2[0]]
            + D1_dnVal[offset1[0] + 1] * D2_dnVal[offset2[0] + 1]
            + D1_dnVal[offset1[0] + 2] * D2_dnVal[offset2[0] + 2]
            + D1_dnVal[offset1[0] + 3] * D2_dnVal[offset2[0] + 3];
            multi[0] *= S_cooVal[eid];
            multi[1] = D1_dnVal[offset1[1]] * D2_dnVal[offset2[1]] 
            + D1_dnVal[offset1[1] + 1] * D2_dnVal[offset2[1] + 1]
            + D1_dnVal[offset1[1] + 2] * D2_dnVal[offset2[1] + 2]
            + D1_dnVal[offset1[1] + 3] * D2_dnVal[offset2[1] + 3];
            multi[1] *= S_cooVal[eid + 1];
            multi[2] = D1_dnVal[offset1[2]] * D2_dnVal[offset2[2]] 
            + D1_dnVal[offset1[2] + 1] * D2_dnVal[offset2[2] + 1]
            + D1_dnVal[offset1[2] + 2] * D2_dnVal[offset2[2] + 2]
            + D1_dnVal[offset1[2] + 3] * D2_dnVal[offset2[2] + 3];
            multi[2] *= S_cooVal[eid + 2];
            multi[3] = D1_dnVal[offset1[3]] * D2_dnVal[offset2[3]] 
            + D1_dnVal[offset1[3] + 1] * D2_dnVal[offset2[3] + 1]
            + D1_dnVal[offset1[3] + 2] * D2_dnVal[offset2[3] + 2]
            + D1_dnVal[offset1[3] + 3] * D2_dnVal[offset2[3] + 3];
            multi[3] *= S_cooVal[eid + 3];
        }
        else
        {
            #pragma unroll 4
            for(int i = 0;i < D_kcols % 4;i++)
            {
                multi[0] = D1_dnVal[offset1[0] + i] * D2_dnVal[offset2[0] + i] * S_cooVal[eid];
                multi[1] = D1_dnVal[offset1[1] + i] * D2_dnVal[offset2[1] + i] * S_cooVal[eid + 1];
                multi[2] = D1_dnVal[offset1[2] + i] * D2_dnVal[offset2[2] + i] * S_cooVal[eid + 2];
                multi[3] = D1_dnVal[offset1[3] + i] * D2_dnVal[offset2[3] + i] * S_cooVal[eid + 3]; 
            }
        }
        atomicAdd(tempoffset, multi[0]);
        atomicAdd(tempoffset + 32, multi[1]);
        atomicAdd(tempoffset + 64, multi[2]);
        atomicAdd(tempoffset + 96, multi[3]);
        __syncthreads();

        if(threadIdx.x < 32)
        {
            T* const tempp = temp + threadIdx.x; 
            multi[0] = *(tempp);
            multi[1] = *(tempp + 32);
            multi[2] = *(tempp + 64);
            multi[3] = *(tempp + 96);
            for (int stride = 16; stride > 0; stride >>= 1)
            {
                multi[0] += __shfl_down_sync(0xffffffff, multi[0], stride, 32);
                multi[1] += __shfl_down_sync(0xffffffff, multi[1], stride, 32);
                multi[2] += __shfl_down_sync(0xffffffff, multi[2], stride, 32);
                multi[3] += __shfl_down_sync(0xffffffff, multi[3], stride, 32);
            }
        }
        if(threadIdx.x == 0)
        {
            *(reinterpret_cast<float4*>(O_cooVal + eid)) = *(reinterpret_cast<float4*>(multi));
        }
    }
    else
    {
        switch (Size % 4)
        {
        case 1:
        {
            offset1[0] = S_cooRowInd[eid] * D_kcols + cid;
            offset2[0] = S_cooColInd[eid] * D_kcols + cid;
            __syncthreads();
            if(cid < D_kcols - 3)
            {
                multi[0] = D1_dnVal[offset1[0]] * D2_dnVal[offset2[0]]
                + D1_dnVal[offset1[0] + 1] * D2_dnVal[offset2[0] + 1]
                + D1_dnVal[offset1[0] + 2] * D2_dnVal[offset2[0] + 2]
                + D1_dnVal[offset1[0] + 3] * D2_dnVal[offset2[0] + 3];
                multi[0] *= S_cooVal[eid];
            }
            else
            {
                #pragma unroll 4
                for(int i = 0;i < D_kcols % 4;i++)
                {
                    multi[0] += D1_dnVal[offset1[0] + i] * D2_dnVal[offset2[0] + i];
                }
                multi[0] *= S_cooVal[eid];
            }
            atomicAdd(&temp[threadIdx.x % 32], multi[0]);
            __syncthreads();
            if(threadIdx.x < 32)
            {
                T* const tempp = temp + threadIdx.x; 
                multi[0] = *tempp;
                for (int stride = 16; stride > 0; stride >>= 1)
                {
                    multi[0] += __shfl_down_sync(0xffffffff, multi[0], stride, 32);
                }
            }
            if(threadIdx.x == 0)             
                O_cooVal[eid] = multi[0];
            break;
        }
        case 2:
        {
            int2 tempRow = *(reinterpret_cast<int2 *> (S_cooRowInd + eid));
            int2 tempCol = *(reinterpret_cast<int2 *> (S_cooColInd + eid));
            offset1[0] = tempRow.x * D_kcols + cid;
            offset1[1] = tempRow.y * D_kcols + cid;
            offset2[0] = tempCol.x * D_kcols + cid;
            offset2[1] = tempCol.y * D_kcols + cid;
            __syncthreads();
            if(cid < D_kcols - 3)
            {
                multi[0] = D1_dnVal[offset1[0]] * D2_dnVal[offset2[0]]
                + D1_dnVal[offset1[0] + 1] * D2_dnVal[offset2[0] + 1]
                + D1_dnVal[offset1[0] + 2] * D2_dnVal[offset2[0] + 2]
                + D1_dnVal[offset1[0] + 3] * D2_dnVal[offset2[0] + 3];
                multi[1] = D1_dnVal[offset1[1]] * D2_dnVal[offset2[1]]
                + D1_dnVal[offset1[1] + 1] * D2_dnVal[offset2[1] + 1]
                + D1_dnVal[offset1[1] + 2] * D2_dnVal[offset2[1] + 2]
                + D1_dnVal[offset1[1] + 3] * D2_dnVal[offset2[1] + 3];
                multi[0] *= S_cooVal[eid];
                multi[1] *= S_cooVal[eid + 1];
            }
            else
            {
                #pragma unroll 4
                for(int i = 0;i < D_kcols % 4;i++)
                {
                    multi[0] += D1_dnVal[offset1[0] + i] * D2_dnVal[offset2[0] + i];
                    multi[1] += D1_dnVal[offset1[1] + i] * D2_dnVal[offset2[1] + i];
                }
                multi[0] *= S_cooVal[eid];
                multi[1] *= S_cooVal[eid + 1];
            }
            atomicAdd(&temp[threadIdx.x % 32], multi[0]);
            atomicAdd(&temp[threadIdx.x % 32 + 32], multi[1]);
            __syncthreads();
            if(threadIdx.x < 32)
            {
                T* const tempp = temp + threadIdx.x; 
                multi[0] = *tempp;
                multi[1] = *(tempp + 32);
                for (int stride = 16; stride > 0; stride >>= 1)
                {
                    multi[0] += __shfl_down_sync(0xffffffff, multi[0], stride, 32);
                    multi[1] += __shfl_down_sync(0xffffffff, multi[1], stride, 32);
                }
            }
            if(threadIdx.x == 0)  
            {
                *(reinterpret_cast<float2*>(O_cooVal + eid)) = *(reinterpret_cast<float2*>(multi));
            }
            break;
        }
        case 3:
        {
            int3 tempRow = *(reinterpret_cast<int3 *> (S_cooRowInd + eid));
            int3 tempCol = *(reinterpret_cast<int3 *> (S_cooColInd + eid));
            offset1[0] = tempRow.x * D_kcols + cid;
            offset1[1] = tempRow.y * D_kcols + cid;
            offset1[2] = tempRow.z * D_kcols + cid;
            offset2[0] = tempCol.x * D_kcols + cid;
            offset2[1] = tempCol.y * D_kcols + cid;
            offset2[2] = tempCol.z * D_kcols + cid;
            __syncthreads();
            if(cid < D_kcols - 3)
            {
                multi[0] = D1_dnVal[offset1[0]] * D2_dnVal[offset2[0]]
                + D1_dnVal[offset1[0] + 1] * D2_dnVal[offset2[0] + 1]
                + D1_dnVal[offset1[0] + 2] * D2_dnVal[offset2[0] + 2]
                + D1_dnVal[offset1[0] + 3] * D2_dnVal[offset2[0] + 3];
                multi[1] = D1_dnVal[offset1[1]] * D2_dnVal[offset2[1]]
                + D1_dnVal[offset1[1] + 1] * D2_dnVal[offset2[1] + 1]
                + D1_dnVal[offset1[1] + 2] * D2_dnVal[offset2[1] + 2]
                + D1_dnVal[offset1[1] + 3] * D2_dnVal[offset2[1] + 3];
                multi[2] = D1_dnVal[offset1[2]] * D2_dnVal[offset2[2]]
                + D1_dnVal[offset1[2] + 1] * D2_dnVal[offset2[2] + 1]
                + D1_dnVal[offset1[2] + 2] * D2_dnVal[offset2[2] + 2]
                + D1_dnVal[offset1[2] + 3] * D2_dnVal[offset2[2] + 3];
                multi[0] *= S_cooVal[eid];
                multi[1] *= S_cooVal[eid + 1];
                multi[2] *= S_cooVal[eid + 2];
            }
            else
            {
                #pragma unroll 4
                for(int i = 0;i < D_kcols % 4;i++)
                {
                    multi[0] += D1_dnVal[offset1[0] + i] * D2_dnVal[offset2[0] + i];
                    multi[1] += D1_dnVal[offset1[1] + i] * D2_dnVal[offset2[1] + i];
                    multi[2] += D1_dnVal[offset1[2] + i] * D2_dnVal[offset2[2] + i];
                }
                multi[0] *= S_cooVal[eid];
                multi[1] *= S_cooVal[eid + 1];
                multi[2] *= S_cooVal[eid + 2];
            }
            atomicAdd(&temp[threadIdx.x % 32], multi[0]);
            atomicAdd(&temp[threadIdx.x % 32 + 32], multi[1]);
            atomicAdd(&temp[threadIdx.x % 32 + 64], multi[2]);
            __syncthreads();
            if(threadIdx.x < 32)
            {
                T* const tempp = temp + threadIdx.x; 
                multi[0] = *tempp;
                multi[1] = *(tempp + 32);
                multi[2] = *(tempp + 64);
                for (int stride = 16; stride > 0; stride >>= 1)
                {
                    multi[0] += __shfl_down_sync(0xffffffff, multi[0], stride, 32);
                    multi[1] += __shfl_down_sync(0xffffffff, multi[1], stride, 32);
                    multi[2] += __shfl_down_sync(0xffffffff, multi[2], stride, 32);
                }
            }
            if(threadIdx.x == 0)  
            {           
                *(reinterpret_cast<float3*>(O_cooVal + eid)) = *(reinterpret_cast<float3*>(multi));
            }
            break;
        }
        default:
            break;
        }
    }
}

template<typename T>
__global__ void sddmmCOOILP8CacheCast(int S_mrows, const int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    __shared__ T temp[256];
    const int eid = (blockIdx.x) << 3;
    const int cid = threadIdx.x;
    temp[cid % 256] = 0;
    __syncthreads();
    int offsetCid = cid % 32;
    float* const tempoffset = temp + offsetCid;
    T multi[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    if (eid < Size - 7)
    {
        int4 tempRow1 = *(reinterpret_cast<int4 *> (S_cooRowInd + eid));
        int4 tempRow2 = *(reinterpret_cast<int4 *> (S_cooRowInd + eid + 4));
        int4 tempCol1 = *(reinterpret_cast<int4 *> (S_cooColInd + eid));
        int4 tempCol2 = *(reinterpret_cast<int4 *> (S_cooColInd + eid + 4));
        __syncthreads();
        multi[0] = D1_dnVal[tempRow1.x * D_kcols + cid] * D2_dnVal[tempCol1.x * D_kcols + cid] * S_cooVal[eid];
        multi[1] = D1_dnVal[tempRow1.y * D_kcols + cid] * D2_dnVal[tempCol1.y * D_kcols + cid] * S_cooVal[eid + 1];
        multi[2] = D1_dnVal[tempRow1.z * D_kcols + cid] * D2_dnVal[tempCol1.z * D_kcols + cid] * S_cooVal[eid + 2];
        multi[3] = D1_dnVal[tempRow1.w * D_kcols + cid] * D2_dnVal[tempCol1.w * D_kcols + cid] * S_cooVal[eid + 3];
        multi[4] = D1_dnVal[tempRow2.x * D_kcols + cid] * D2_dnVal[tempCol2.x * D_kcols + cid] * S_cooVal[eid + 4];
        multi[5] = D1_dnVal[tempRow2.y * D_kcols + cid] * D2_dnVal[tempCol2.y * D_kcols + cid] * S_cooVal[eid + 5];
        multi[6] = D1_dnVal[tempRow2.z * D_kcols + cid] * D2_dnVal[tempCol2.z * D_kcols + cid] * S_cooVal[eid + 6];
        multi[7] = D1_dnVal[tempRow2.w * D_kcols + cid] * D2_dnVal[tempCol2.w * D_kcols + cid] * S_cooVal[eid + 7];

        atomicAdd(tempoffset, multi[0]);
        atomicAdd(tempoffset + 32, multi[1]);
        atomicAdd(tempoffset + 64, multi[2]);
        atomicAdd(tempoffset + 96, multi[3]); 
        atomicAdd(tempoffset + 128, multi[4]);
        atomicAdd(tempoffset + 160, multi[5]);
        atomicAdd(tempoffset + 192, multi[6]);
        atomicAdd(tempoffset + 224, multi[7]);
        __syncthreads();
        if(threadIdx.x < 32)
        {
            float* const tempp = temp + threadIdx.x;
            multi[0] = *tempp;
            multi[1] = *(tempp + 32);
            multi[2] = *(tempp + 64);
            multi[3] = *(tempp + 96);
            multi[4] = *(tempp + 128);
            multi[5] = *(tempp + 160);
            multi[6] = *(tempp + 192);
            multi[7] = *(tempp + 224);
            for (int stride = 16; stride > 0; stride >>= 1)
            {
                multi[0] += __shfl_down_sync(0xffffffff, multi[0], stride, 32);
                multi[1] += __shfl_down_sync(0xffffffff, multi[1], stride, 32);
                multi[2] += __shfl_down_sync(0xffffffff, multi[2], stride, 32);
                multi[3] += __shfl_down_sync(0xffffffff, multi[3], stride, 32);
                multi[4] += __shfl_down_sync(0xffffffff, multi[4], stride, 32);
                multi[5] += __shfl_down_sync(0xffffffff, multi[5], stride, 32);
                multi[6] += __shfl_down_sync(0xffffffff, multi[6], stride, 32);
                multi[7] += __shfl_down_sync(0xffffffff, multi[7], stride, 32);
            }
        }
        if(threadIdx.x == 0)
        {
            *(reinterpret_cast<float4*>(O_cooVal + eid)) = 
            *(reinterpret_cast<float4*>(multi));
            *(reinterpret_cast<float4*>(O_cooVal + eid + 4)) =
            *(reinterpret_cast<float4*>(multi + 4));
        }
    }
    else
    {
        int offset1[8], offset2[8];
        switch (Size % 8)
        {
        case 1:
        {
            offset1[0] = S_cooRowInd[eid] * D_kcols;
            offset2[0] = S_cooColInd[eid] * D_kcols;
            multi[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
            atomicAdd(&O_cooVal[eid], multi[0]);
            break;
        }
        case 2:
        {
            offset1[0] = S_cooRowInd[eid] * D_kcols;
            offset1[1] = S_cooRowInd[eid + 1] * D_kcols;

            offset2[0] = S_cooColInd[eid] * D_kcols;
            offset2[1] = S_cooColInd[eid + 1] * D_kcols;

            multi[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
            multi[1] = D1_dnVal[offset1[1] + cid] * D2_dnVal[offset2[1] + cid] * S_cooVal[eid + 1];

            atomicAdd(&O_cooVal[eid], multi[0]);
            atomicAdd(&O_cooVal[eid + 1], multi[1]);
            break;
        }
        case 3:
        {
            offset1[0] = S_cooRowInd[eid] * D_kcols;
            offset1[1] = S_cooRowInd[eid + 1] * D_kcols;
            offset1[2] = S_cooRowInd[eid + 2] * D_kcols;

            offset2[0] = S_cooColInd[eid] * D_kcols;
            offset2[1] = S_cooColInd[eid + 1] * D_kcols;
            offset2[2] = S_cooColInd[eid + 2] * D_kcols;

            multi[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
            multi[1] = D1_dnVal[offset1[1] + cid] * D2_dnVal[offset2[1] + cid] * S_cooVal[eid + 1];
            multi[2] = D1_dnVal[offset1[2] + cid] * D2_dnVal[offset2[2] + cid] * S_cooVal[eid + 2];

            atomicAdd(&O_cooVal[eid], multi[0]);
            atomicAdd(&O_cooVal[eid + 1], multi[1]);
            atomicAdd(&O_cooVal[eid + 2], multi[2]);
            break;
        }
        case 4:
        {
            offset1[0] = S_cooRowInd[eid] * D_kcols;
            offset1[1] = S_cooRowInd[eid + 1] * D_kcols;
            offset1[2] = S_cooRowInd[eid + 2] * D_kcols;
            offset1[3] = S_cooRowInd[eid + 3] * D_kcols;

            offset2[0] = S_cooColInd[eid] * D_kcols;
            offset2[1] = S_cooColInd[eid + 1] * D_kcols;
            offset2[2] = S_cooColInd[eid + 2] * D_kcols;
            offset2[3] = S_cooColInd[eid + 3] * D_kcols;

            multi[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
            multi[1] = D1_dnVal[offset1[1] + cid] * D2_dnVal[offset2[1] + cid] * S_cooVal[eid + 1];
            multi[2] = D1_dnVal[offset1[2] + cid] * D2_dnVal[offset2[2] + cid] * S_cooVal[eid + 2];
            multi[3] = D1_dnVal[offset1[3] + cid] * D2_dnVal[offset2[3] + cid] * S_cooVal[eid + 3];

            atomicAdd(&O_cooVal[eid], multi[0]);
            atomicAdd(&O_cooVal[eid + 1], multi[1]);
            atomicAdd(&O_cooVal[eid + 2], multi[2]);
            atomicAdd(&O_cooVal[eid + 3], multi[3]);
            break;
        }
        case 5:
        {
            offset1[0] = S_cooRowInd[eid] * D_kcols;
            offset1[1] = S_cooRowInd[eid + 1] * D_kcols;
            offset1[2] = S_cooRowInd[eid + 2] * D_kcols;
            offset1[3] = S_cooRowInd[eid + 3] * D_kcols;
            offset1[4] = S_cooRowInd[eid + 4] * D_kcols;

            offset2[0] = S_cooColInd[eid] * D_kcols;
            offset2[1] = S_cooColInd[eid + 1] * D_kcols;
            offset2[2] = S_cooColInd[eid + 2] * D_kcols;
            offset2[3] = S_cooColInd[eid + 3] * D_kcols;
            offset2[4] = S_cooColInd[eid + 4] * D_kcols;

            multi[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
            multi[1] = D1_dnVal[offset1[1] + cid] * D2_dnVal[offset2[1] + cid] * S_cooVal[eid + 1];
            multi[2] = D1_dnVal[offset1[2] + cid] * D2_dnVal[offset2[2] + cid] * S_cooVal[eid + 2];
            multi[3] = D1_dnVal[offset1[3] + cid] * D2_dnVal[offset2[3] + cid] * S_cooVal[eid + 3];
            multi[4] = D1_dnVal[offset1[4] + cid] * D2_dnVal[offset2[4] + cid] * S_cooVal[eid + 4];

            atomicAdd(&O_cooVal[eid], multi[0]);
            atomicAdd(&O_cooVal[eid + 1], multi[1]);
            atomicAdd(&O_cooVal[eid + 2], multi[2]);
            atomicAdd(&O_cooVal[eid + 3], multi[3]);
            atomicAdd(&O_cooVal[eid + 4], multi[4]);
            break;
        }
        case 6:
        {
            offset1[0] = S_cooRowInd[eid] * D_kcols;
            offset1[1] = S_cooRowInd[eid + 1] * D_kcols;
            offset1[2] = S_cooRowInd[eid + 2] * D_kcols;
            offset1[3] = S_cooRowInd[eid + 3] * D_kcols;
            offset1[4] = S_cooRowInd[eid + 4] * D_kcols;
            offset1[5] = S_cooRowInd[eid + 5] * D_kcols;

            offset2[1] = S_cooColInd[eid + 1] * D_kcols;
            offset2[2] = S_cooColInd[eid + 2] * D_kcols;
            offset2[3] = S_cooColInd[eid + 3] * D_kcols;
            offset2[4] = S_cooColInd[eid + 4] * D_kcols;
            offset2[5] = S_cooColInd[eid + 5] * D_kcols;

            multi[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
            multi[1] = D1_dnVal[offset1[1] + cid] * D2_dnVal[offset2[1] + cid] * S_cooVal[eid + 1];
            multi[2] = D1_dnVal[offset1[2] + cid] * D2_dnVal[offset2[2] + cid] * S_cooVal[eid + 2];
            multi[3] = D1_dnVal[offset1[3] + cid] * D2_dnVal[offset2[3] + cid] * S_cooVal[eid + 3];
            multi[4] = D1_dnVal[offset1[4] + cid] * D2_dnVal[offset2[4] + cid] * S_cooVal[eid + 4];
            multi[5] = D1_dnVal[offset1[5] + cid] * D2_dnVal[offset2[5] + cid] * S_cooVal[eid + 5];

            atomicAdd(&O_cooVal[eid], multi[0]);
            atomicAdd(&O_cooVal[eid + 1], multi[1]);
            atomicAdd(&O_cooVal[eid + 2], multi[2]);
            atomicAdd(&O_cooVal[eid + 3], multi[3]);
            atomicAdd(&O_cooVal[eid + 4], multi[4]);
            atomicAdd(&O_cooVal[eid + 5], multi[5]);

            break;
        }
        case 7:
        {
            offset1[0] = S_cooRowInd[eid] * D_kcols;
            offset1[1] = S_cooRowInd[eid + 1] * D_kcols;
            offset1[2] = S_cooRowInd[eid + 2] * D_kcols;
            offset1[3] = S_cooRowInd[eid + 3] * D_kcols;
            offset1[4] = S_cooRowInd[eid + 4] * D_kcols;
            offset1[5] = S_cooRowInd[eid + 5] * D_kcols;
            offset1[6] = S_cooRowInd[eid + 6] * D_kcols;

            offset2[0] = S_cooColInd[eid] * D_kcols;
            offset2[1] = S_cooColInd[eid + 1] * D_kcols;
            offset2[2] = S_cooColInd[eid + 2] * D_kcols;
            offset2[3] = S_cooColInd[eid + 3] * D_kcols;
            offset2[4] = S_cooColInd[eid + 4] * D_kcols;
            offset2[5] = S_cooColInd[eid + 5] * D_kcols;
            offset2[6] = S_cooColInd[eid + 6] * D_kcols;

            multi[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
            multi[1] = D1_dnVal[offset1[1] + cid] * D2_dnVal[offset2[1] + cid] * S_cooVal[eid + 1];
            multi[2] = D1_dnVal[offset1[2] + cid] * D2_dnVal[offset2[2] + cid] * S_cooVal[eid + 2];
            multi[3] = D1_dnVal[offset1[3] + cid] * D2_dnVal[offset2[3] + cid] * S_cooVal[eid + 3];
            multi[4] = D1_dnVal[offset1[4] + cid] * D2_dnVal[offset2[4] + cid] * S_cooVal[eid + 4];
            multi[5] = D1_dnVal[offset1[5] + cid] * D2_dnVal[offset2[5] + cid] * S_cooVal[eid + 5];
            multi[6] = D1_dnVal[offset1[6] + cid] * D2_dnVal[offset2[6] + cid] * S_cooVal[eid + 6];

            atomicAdd(&O_cooVal[eid], multi[0]);
            atomicAdd(&O_cooVal[eid + 1], multi[1]);
            atomicAdd(&O_cooVal[eid + 2], multi[2]);
            atomicAdd(&O_cooVal[eid + 3], multi[3]);
            atomicAdd(&O_cooVal[eid + 4], multi[4]);
            atomicAdd(&O_cooVal[eid + 5], multi[5]);
            atomicAdd(&O_cooVal[eid + 6], multi[6]);
            break;
        }
        default:
            break;
        }
    }
}

template<typename T>
__global__ void sddmmCOOILP8CacheCastNext(int S_mrows, const int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    __shared__ T temp[256];
    int eid = (blockIdx.x) << 3;
    int cid = (threadIdx.x << 1);
    temp[threadIdx.x % 256] = 0;
    __syncthreads();
    int offsetCid = threadIdx.x % 32;
    T* const tempoffset = temp + offsetCid;
    T multi[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    if (eid < Size - 7)
    {
        int4 tempRow1 = *(reinterpret_cast<int4 *> (S_cooRowInd + eid));
        int4 tempRow2 = *(reinterpret_cast<int4 *> (S_cooRowInd + eid + 4));
        int4 tempCol1 = *(reinterpret_cast<int4 *> (S_cooColInd + eid));
        int4 tempCol2 = *(reinterpret_cast<int4 *> (S_cooColInd + eid + 4));
        tempRow1.x = tempRow1.x * D_kcols + cid;
        tempRow1.y = tempRow1.y * D_kcols + cid;
        tempRow1.z = tempRow1.z * D_kcols + cid;
        tempRow1.w = tempRow1.w * D_kcols + cid;
        tempRow2.x = tempRow2.x * D_kcols + cid;
        tempRow2.y = tempRow2.y * D_kcols + cid;
        tempRow2.z = tempRow2.z * D_kcols + cid;
        tempRow2.w = tempRow2.w * D_kcols + cid;
        tempCol1.x = tempCol1.x * D_kcols + cid;
        tempCol1.y = tempCol1.y * D_kcols + cid;
        tempCol1.z = tempCol1.z * D_kcols + cid;
        tempCol1.w = tempCol1.w * D_kcols + cid;
        tempCol2.x = tempCol2.x * D_kcols + cid;
        tempCol2.y = tempCol2.y * D_kcols + cid;
        tempCol2.z = tempCol2.z * D_kcols + cid;
        tempCol2.w = tempCol2.w * D_kcols + cid;
        __syncthreads();
        if(cid < D_kcols - 1)
        {
            multi[0] = D1_dnVal[tempRow1.x] * D2_dnVal[tempCol1.x] 
            + D1_dnVal[tempRow1.x + 1] * D2_dnVal[tempCol1.x + 1]; 
            multi[0] *= S_cooVal[eid];
            multi[1] = D1_dnVal[tempRow1.y] * D2_dnVal[tempCol1.y]
            + D1_dnVal[tempRow1.y + 1] * D2_dnVal[tempCol1.y + 1];
            multi[1] *= S_cooVal[eid + 1];
            multi[2] = D1_dnVal[tempRow1.z] * D2_dnVal[tempCol1.z]
            + D1_dnVal[tempRow1.z + 1] * D2_dnVal[tempCol1.z + 1];
            multi[2] *= S_cooVal[eid + 2];
            multi[3] = D1_dnVal[tempRow1.w] * D2_dnVal[tempCol1.w]
            + D1_dnVal[tempRow1.w + 1] * D2_dnVal[tempCol1.w + 1]; 
            multi[3] *= S_cooVal[eid + 3];
            multi[4] = D1_dnVal[tempRow2.x] * D2_dnVal[tempCol2.x]
            + D1_dnVal[tempRow2.x + 1] * D2_dnVal[tempCol2.x + 1]; 
            multi[4] *= S_cooVal[eid + 4];
            multi[5] = D1_dnVal[tempRow2.y] * D2_dnVal[tempCol2.y]
            + D1_dnVal[tempRow2.y + 1] * D2_dnVal[tempCol2.y + 1]; 
            multi[5] *= S_cooVal[eid + 5];
            multi[6] = D1_dnVal[tempRow2.z] * D2_dnVal[tempCol2.z]
            + D1_dnVal[tempRow2.z + 1] * D2_dnVal[tempCol2.z + 1];
            multi[6] *= S_cooVal[eid + 6];
            multi[7] = D1_dnVal[tempRow2.w] * D2_dnVal[tempCol2.w] 
            + D1_dnVal[tempRow2.w + 1] * D2_dnVal[tempCol2.w + 1]; 
            multi[7] *= S_cooVal[eid + 7];
        }
        else
        {
            multi[0] = D1_dnVal[tempRow1.x] * D2_dnVal[tempCol1.x] * S_cooVal[eid];
            multi[1] = D1_dnVal[tempRow1.y] * D2_dnVal[tempCol1.y] * S_cooVal[eid + 1];
            multi[2] = D1_dnVal[tempRow1.z] * D2_dnVal[tempCol1.z] * S_cooVal[eid + 2];
            multi[3] = D1_dnVal[tempRow1.w] * D2_dnVal[tempCol1.w] * S_cooVal[eid + 3];
            multi[4] = D1_dnVal[tempRow2.x] * D2_dnVal[tempCol2.x] * S_cooVal[eid + 4];
            multi[5] = D1_dnVal[tempRow2.y] * D2_dnVal[tempCol2.y] * S_cooVal[eid + 5];
            multi[6] = D1_dnVal[tempRow2.z] * D2_dnVal[tempCol2.z] * S_cooVal[eid + 6];
            multi[7] = D1_dnVal[tempRow2.w] * D2_dnVal[tempCol2.w] * S_cooVal[eid + 7]; 
        }
        atomicAdd(tempoffset, multi[0]);
        atomicAdd(tempoffset + 32, multi[1]);
        atomicAdd(tempoffset + 64, multi[2]);
        atomicAdd(tempoffset + 96, multi[3]); 
        atomicAdd(tempoffset + 128, multi[4]);
        atomicAdd(tempoffset + 160, multi[5]);
        atomicAdd(tempoffset + 192, multi[6]);
        atomicAdd(tempoffset + 224, multi[7]);
        __syncthreads();
        if(threadIdx.x < 32)
        {
            T* const tempp = temp + threadIdx.x;
            multi[0] = *tempp;
            multi[1] = *(tempp + 32);
            multi[2] = *(tempp + 64);
            multi[3] = *(tempp + 96);
            multi[4] = *(tempp + 128);
            multi[5] = *(tempp + 160);
            multi[6] = *(tempp + 192);
            multi[7] = *(tempp + 224);
            for (int stride = 16; stride > 0; stride >>= 1)
            {
                multi[0] += __shfl_down_sync(0xffffffff, multi[0], stride, 32);
                multi[1] += __shfl_down_sync(0xffffffff, multi[1], stride, 32);
                multi[2] += __shfl_down_sync(0xffffffff, multi[2], stride, 32);
                multi[3] += __shfl_down_sync(0xffffffff, multi[3], stride, 32);
                multi[4] += __shfl_down_sync(0xffffffff, multi[4], stride, 32);
                multi[5] += __shfl_down_sync(0xffffffff, multi[5], stride, 32);
                multi[6] += __shfl_down_sync(0xffffffff, multi[6], stride, 32);
                multi[7] += __shfl_down_sync(0xffffffff, multi[7], stride, 32);
            }
        }
        if(threadIdx.x == 0)
        {
            *(reinterpret_cast<float4*>(O_cooVal + eid)) = *(reinterpret_cast<float4*>(multi));
            *(reinterpret_cast<float4*>(O_cooVal + eid + 4)) = *(reinterpret_cast<float4*>(multi + 4));
        }
    }
    else
    {
        int offset1[8], offset2[8];
        if(cid < D_kcols - 1)
        {
            for(int i = 0;i < (Size % 8);i++)
            {
                offset1[i] = S_cooRowInd[eid + i] * D_kcols;
                offset2[i] = S_cooColInd[eid + i] * D_kcols;
                multi[i] = D1_dnVal[offset1[i] + cid] * D2_dnVal[offset2[i] + cid]
                + D1_dnVal[offset1[i] + cid + 1] * D2_dnVal[offset2[i] + cid + 1]; 
                atomicAdd(&O_cooVal[eid + i], S_cooVal[eid + i] * multi[i]);
            }
        }
        else
        {
           for(int i = 0;i < (Size % 8);i++)
            {
                offset1[i] = S_cooRowInd[eid + i] * D_kcols;
                offset2[i] = S_cooColInd[eid + i] * D_kcols;
                multi[i] = D1_dnVal[offset1[i] + cid] * D2_dnVal[offset2[i] + cid];
                atomicAdd(&O_cooVal[eid + i], S_cooVal[eid + i] * multi[i]);
            } 
        }
    }
}

template<typename T>
__global__ void sddmmCOOILP8CacheCastNext4(int S_mrows, const int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    __shared__ T temp[256];
    const int eid = (blockIdx.x) << 3;
    const int cid = (threadIdx.x << 2);
    temp[(threadIdx.x << 1) % 256] = 0;
    temp[((threadIdx.x << 1) + 1) % 256] = 0;
    __syncthreads();
    int offsetCid = threadIdx.x % 32;
    T* const tempoffset = temp + offsetCid;
    T multi[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    if (eid < Size - 7)
    {
        int4 tempRow1 = *(reinterpret_cast<int4 *> (S_cooRowInd + eid));
        int4 tempRow2 = *(reinterpret_cast<int4 *> (S_cooRowInd + eid + 4));
        int4 tempCol1 = *(reinterpret_cast<int4 *> (S_cooColInd + eid));
        int4 tempCol2 = *(reinterpret_cast<int4 *> (S_cooColInd + eid + 4));
        tempRow1.x = tempRow1.x * D_kcols + cid;
        tempRow1.y = tempRow1.y * D_kcols + cid;
        tempRow1.z = tempRow1.z * D_kcols + cid;
        tempRow1.w = tempRow1.w * D_kcols + cid;
        tempRow2.x = tempRow2.x * D_kcols + cid;
        tempRow2.y = tempRow2.y * D_kcols + cid;
        tempRow2.z = tempRow2.z * D_kcols + cid;
        tempRow2.w = tempRow2.w * D_kcols + cid;
        tempCol1.x = tempCol1.x * D_kcols + cid;
        tempCol1.y = tempCol1.y * D_kcols + cid;
        tempCol1.z = tempCol1.z * D_kcols + cid;
        tempCol1.w = tempCol1.w * D_kcols + cid;
        tempCol2.x = tempCol2.x * D_kcols + cid;
        tempCol2.y = tempCol2.y * D_kcols + cid;
        tempCol2.z = tempCol2.z * D_kcols + cid;
        tempCol2.w = tempCol2.w * D_kcols + cid;
        __syncthreads();
        if(cid < D_kcols - 3)
        {
            multi[0] = D1_dnVal[tempRow1.x] * D2_dnVal[tempCol1.x] 
            + D1_dnVal[tempRow1.x + 1] * D2_dnVal[tempCol1.x + 1]
            + D1_dnVal[tempRow1.x + 2] * D2_dnVal[tempCol1.x + 2]
            + D1_dnVal[tempRow1.x + 3] * D2_dnVal[tempCol1.x + 3]; 
            multi[0] *= S_cooVal[eid];
            multi[1] = D1_dnVal[tempRow1.y] * D2_dnVal[tempCol1.y]
            + D1_dnVal[tempRow1.y + 1] * D2_dnVal[tempCol1.y + 1]
            + D1_dnVal[tempRow1.y + 2] * D2_dnVal[tempCol1.y + 2]
            + D1_dnVal[tempRow1.y + 3] * D2_dnVal[tempCol1.y + 3];
            multi[1] *= S_cooVal[eid + 1];
            multi[2] = D1_dnVal[tempRow1.z] * D2_dnVal[tempCol1.z]
            + D1_dnVal[tempRow1.z + 1] * D2_dnVal[tempCol1.z + 1]
            + D1_dnVal[tempRow1.z + 2] * D2_dnVal[tempCol1.z + 2]
            + D1_dnVal[tempRow1.z + 3] * D2_dnVal[tempCol1.z + 3];
            multi[2] *= S_cooVal[eid + 2];
            multi[3] = D1_dnVal[tempRow1.w] * D2_dnVal[tempCol1.w]
            + D1_dnVal[tempRow1.w + 1] * D2_dnVal[tempCol1.w + 1]
            + D1_dnVal[tempRow1.w + 2] * D2_dnVal[tempCol1.w + 2]
            + D1_dnVal[tempRow1.w + 3] * D2_dnVal[tempCol1.w + 3]; 
            multi[3] *= S_cooVal[eid + 3];
            multi[4] = D1_dnVal[tempRow2.x] * D2_dnVal[tempCol2.x]
            + D1_dnVal[tempRow2.x + 1] * D2_dnVal[tempCol2.x + 1]
            + D1_dnVal[tempRow2.x + 2] * D2_dnVal[tempCol2.x + 2]
            + D1_dnVal[tempRow2.x + 3] * D2_dnVal[tempCol2.x + 3]; 
            multi[4] *= S_cooVal[eid + 4];
            multi[5] = D1_dnVal[tempRow2.y] * D2_dnVal[tempCol2.y]
            + D1_dnVal[tempRow2.y + 1] * D2_dnVal[tempCol2.y + 1]
            + D1_dnVal[tempRow2.y + 2] * D2_dnVal[tempCol2.y + 2]
            + D1_dnVal[tempRow2.y + 3] * D2_dnVal[tempCol2.y + 3]; 
            multi[5] *= S_cooVal[eid + 5];
            multi[6] = D1_dnVal[tempRow2.z] * D2_dnVal[tempCol2.z]
            + D1_dnVal[tempRow2.z + 1] * D2_dnVal[tempCol2.z + 1]
            + D1_dnVal[tempRow2.z + 2] * D2_dnVal[tempCol2.z + 2]
            + D1_dnVal[tempRow2.z + 3] * D2_dnVal[tempCol2.z + 3];
            multi[6] *= S_cooVal[eid + 6];
            multi[7] = D1_dnVal[tempRow2.w] * D2_dnVal[tempCol2.w] 
            + D1_dnVal[tempRow2.w + 1] * D2_dnVal[tempCol2.w + 1]
            + D1_dnVal[tempRow2.w + 2] * D2_dnVal[tempCol2.w + 2]
            + D1_dnVal[tempRow2.w + 3] * D2_dnVal[tempCol2.w + 3]; 
            multi[7] *= S_cooVal[eid + 7];
        }
        else
        {
            for(int i = 0;i < D_kcols % 4;i++)
            {
                multi[0] = D1_dnVal[tempRow1.x + i] * D2_dnVal[tempRow2.x + i] * S_cooVal[eid];
                multi[1] = D1_dnVal[tempRow1.y + i] * D2_dnVal[tempRow2.y + i] * S_cooVal[eid + 1];
                multi[2] = D1_dnVal[tempRow1.z + i] * D2_dnVal[tempRow2.z + i] * S_cooVal[eid + 2];
                multi[3] = D1_dnVal[tempRow1.w + i] * D2_dnVal[tempRow2.w + i] * S_cooVal[eid + 3]; 
            }
        }
        atomicAdd(tempoffset, multi[0]);
        atomicAdd(tempoffset + 32, multi[1]);
        atomicAdd(tempoffset + 64, multi[2]);
        atomicAdd(tempoffset + 96, multi[3]); 
        atomicAdd(tempoffset + 128, multi[4]);
        atomicAdd(tempoffset + 160, multi[5]);
        atomicAdd(tempoffset + 192, multi[6]);
        atomicAdd(tempoffset + 224, multi[7]);
        __syncthreads();
        if(threadIdx.x < 32)
        {
            T* const tempp = temp + threadIdx.x;
            multi[0] = *tempp;
            multi[1] = *(tempp + 32);
            multi[2] = *(tempp + 64);
            multi[3] = *(tempp + 96);
            multi[4] = *(tempp + 128);
            multi[5] = *(tempp + 160);
            multi[6] = *(tempp + 192);
            multi[7] = *(tempp + 224);
            for (int stride = 16; stride > 0; stride >>= 1)
            {
                multi[0] += __shfl_down_sync(0xffffffff, multi[0], stride, 32);
                multi[1] += __shfl_down_sync(0xffffffff, multi[1], stride, 32);
                multi[2] += __shfl_down_sync(0xffffffff, multi[2], stride, 32);
                multi[3] += __shfl_down_sync(0xffffffff, multi[3], stride, 32);
                multi[4] += __shfl_down_sync(0xffffffff, multi[4], stride, 32);
                multi[5] += __shfl_down_sync(0xffffffff, multi[5], stride, 32);
                multi[6] += __shfl_down_sync(0xffffffff, multi[6], stride, 32);
                multi[7] += __shfl_down_sync(0xffffffff, multi[7], stride, 32);
            }
        }
        if(threadIdx.x == 0)
        {
            *(reinterpret_cast<float4*>(O_cooVal + eid)) = *(reinterpret_cast<float4*>(multi));
            *(reinterpret_cast<float4*>(O_cooVal + eid + 4)) = *(reinterpret_cast<float4*>(multi + 4));
        }
    }
    else
    {
        int offset1[8], offset2[8];
        if(cid < D_kcols - 3)
        {
            for(int i = 0;i < (Size % 8);i++)
            {
                offset1[i] = S_cooRowInd[eid + i] * D_kcols;
                offset2[i] = S_cooColInd[eid + i] * D_kcols;
                for(int j = 0;j < 4;j++)
                {
                    multi[i] += D1_dnVal[offset1[i] + cid + j] * D2_dnVal[offset2[i] + cid + j];
                }
                atomicAdd(&O_cooVal[eid + i], S_cooVal[eid + i] * multi[i]);
            }
        }
        else
        {
           for(int i = 0;i < (Size % 8);i++)
            {
                offset1[i] = S_cooRowInd[eid + i] * D_kcols;
                offset2[i] = S_cooColInd[eid + i] * D_kcols;
                for(int j = 0;j < (D_kcols % 4);j++)
                {
                    multi[i] += D1_dnVal[offset1[i] + cid + j] * D2_dnVal[offset2[i] + cid + j];
                }
                atomicAdd(&O_cooVal[eid + i], S_cooVal[eid + i] * multi[i]);
            } 
        }
    }
}

//deprecate
template<typename T>
__global__ void sddmmCOOILP16CacheCast(int S_mrows, const int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    __shared__ T temp[512];
    const int eid = (blockIdx.x) << 4;
    const int cid = threadIdx.x;
    temp[(cid % 256) << 1] = 0;
    temp[((cid % 256) << 1) + 1] = 0;
    __syncthreads();
    T multi[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    if (eid < Size - 15)
    {
        int4 tempRow1 = *(reinterpret_cast<int4 *> (S_cooRowInd + eid));
        int4 tempCol1 = *(reinterpret_cast<int4 *> (S_cooColInd + eid));
        multi[0] = D1_dnVal[tempRow1.x * D_kcols + cid] * D2_dnVal[tempCol1.x * D_kcols + cid] * S_cooVal[eid];
        multi[1] = D1_dnVal[tempRow1.y * D_kcols + cid] * D2_dnVal[tempCol1.y * D_kcols + cid] * S_cooVal[eid + 1];
        multi[2] = D1_dnVal[tempRow1.z * D_kcols + cid] * D2_dnVal[tempCol1.z * D_kcols + cid] * S_cooVal[eid + 2];
        multi[3] = D1_dnVal[tempRow1.w * D_kcols + cid] * D2_dnVal[tempCol1.w * D_kcols + cid] * S_cooVal[eid + 3];
        tempRow1 = *(reinterpret_cast<int4 *> (S_cooRowInd + eid + 4));
        tempCol1 = *(reinterpret_cast<int4 *> (S_cooColInd + eid + 4));
        multi[4] = D1_dnVal[tempRow1.x * D_kcols + cid] * D2_dnVal[tempCol1.x * D_kcols + cid] * S_cooVal[eid + 4];
        multi[5] = D1_dnVal[tempRow1.y * D_kcols + cid] * D2_dnVal[tempCol1.y * D_kcols + cid] * S_cooVal[eid + 5];
        multi[6] = D1_dnVal[tempRow1.z * D_kcols + cid] * D2_dnVal[tempCol1.z * D_kcols + cid] * S_cooVal[eid + 6];
        multi[7] = D1_dnVal[tempRow1.w * D_kcols + cid] * D2_dnVal[tempCol1.w * D_kcols + cid] * S_cooVal[eid + 7];
        tempRow1 = *(reinterpret_cast<int4 *> (S_cooRowInd + eid + 8));
        tempCol1 = *(reinterpret_cast<int4 *> (S_cooColInd + eid + 8));
        multi[8] = D1_dnVal[tempRow1.x * D_kcols + cid] * D2_dnVal[tempCol1.x * D_kcols + cid] * S_cooVal[eid + 8];
        multi[9] = D1_dnVal[tempRow1.y * D_kcols + cid] * D2_dnVal[tempCol1.y * D_kcols + cid] * S_cooVal[eid + 9];
        multi[10] = D1_dnVal[tempRow1.z * D_kcols + cid] * D2_dnVal[tempCol1.z * D_kcols + cid] * S_cooVal[eid + 10];
        multi[11] = D1_dnVal[tempRow1.w * D_kcols + cid] * D2_dnVal[tempCol1.w * D_kcols + cid] * S_cooVal[eid + 11];
        tempRow1 = *(reinterpret_cast<int4 *> (S_cooRowInd + eid + 12));
        tempCol1 = *(reinterpret_cast<int4 *> (S_cooColInd + eid + 12));
        multi[12] = D1_dnVal[tempRow1.x * D_kcols + cid] * D2_dnVal[tempCol1.x * D_kcols + cid] * S_cooVal[eid + 12];
        multi[13] = D1_dnVal[tempRow1.y * D_kcols + cid] * D2_dnVal[tempCol1.y * D_kcols + cid] * S_cooVal[eid + 13];
        multi[14] = D1_dnVal[tempRow1.z * D_kcols + cid] * D2_dnVal[tempCol1.z * D_kcols + cid] * S_cooVal[eid + 14];
        multi[15] = D1_dnVal[tempRow1.w * D_kcols + cid] * D2_dnVal[tempCol1.w * D_kcols + cid] * S_cooVal[eid + 15];

        __syncthreads();
        atomicAdd(&temp[cid % 32], multi[0]);
        atomicAdd(&temp[cid % 32 + 32], multi[1]);
        atomicAdd(&temp[cid % 32 + 64], multi[2]);
        atomicAdd(&temp[cid % 32 + 96], multi[3]); 
        atomicAdd(&temp[cid % 32 + 128], multi[4]);
        atomicAdd(&temp[cid % 32 + 160], multi[5]);
        atomicAdd(&temp[cid % 32 + 192], multi[6]);
        atomicAdd(&temp[cid % 32 + 224], multi[7]);
        atomicAdd(&temp[cid % 32 + 256], multi[8]);
        atomicAdd(&temp[cid % 32 + 288], multi[9]);
        atomicAdd(&temp[cid % 32 + 320], multi[10]);
        atomicAdd(&temp[cid % 32 + 352], multi[11]);
        atomicAdd(&temp[cid % 32 + 384], multi[12]); 
        atomicAdd(&temp[cid % 32 + 416], multi[13]);
        atomicAdd(&temp[cid % 32 + 448], multi[14]);
        atomicAdd(&temp[cid % 32 + 480], multi[15]);
        __syncthreads();
        if(threadIdx.x < 32)
        {
            multi[0] = temp[threadIdx.x];
            multi[1] = temp[threadIdx.x + 32];
            multi[2] = temp[threadIdx.x + 64];
            multi[3] = temp[threadIdx.x + 96];
            multi[4] = temp[threadIdx.x + 128];
            multi[5] = temp[threadIdx.x + 160];
            multi[6] = temp[threadIdx.x + 192];
            multi[7] = temp[threadIdx.x + 224];
            multi[8] = temp[threadIdx.x + 256];
            multi[9] = temp[threadIdx.x + 288];
            multi[10] = temp[threadIdx.x + 320];
            multi[11] = temp[threadIdx.x + 352];
            multi[12] = temp[threadIdx.x + 384];
            multi[13] = temp[threadIdx.x + 416];
            multi[14] = temp[threadIdx.x + 448];
            multi[15] = temp[threadIdx.x + 480];
            for (int stride = 16; stride > 0; stride >>= 1)
            {
                multi[0] += __shfl_down_sync(0xffffffff, multi[0], stride, 32);
                multi[1] += __shfl_down_sync(0xffffffff, multi[1], stride, 32);
                multi[2] += __shfl_down_sync(0xffffffff, multi[2], stride, 32);
                multi[3] += __shfl_down_sync(0xffffffff, multi[3], stride, 32);
                multi[4] += __shfl_down_sync(0xffffffff, multi[4], stride, 32);
                multi[5] += __shfl_down_sync(0xffffffff, multi[5], stride, 32);
                multi[6] += __shfl_down_sync(0xffffffff, multi[6], stride, 32);
                multi[7] += __shfl_down_sync(0xffffffff, multi[7], stride, 32);
                multi[8] += __shfl_down_sync(0xffffffff, multi[8], stride, 32);
                multi[9] += __shfl_down_sync(0xffffffff, multi[9], stride, 32);
                multi[10] += __shfl_down_sync(0xffffffff, multi[10], stride, 32);
                multi[11] += __shfl_down_sync(0xffffffff, multi[11], stride, 32);
                multi[12] += __shfl_down_sync(0xffffffff, multi[12], stride, 32);
                multi[13] += __shfl_down_sync(0xffffffff, multi[13], stride, 32);
                multi[14] += __shfl_down_sync(0xffffffff, multi[14], stride, 32);
                multi[15] += __shfl_down_sync(0xffffffff, multi[15], stride, 32);
            }
        }
        if(threadIdx.x == 0)
        {
            *(reinterpret_cast<float4*>(O_cooVal + eid)) = *(reinterpret_cast<float4*>(multi));
            *(reinterpret_cast<float4*>(O_cooVal + eid + 4)) = *(reinterpret_cast<float4*>(multi + 4));
            *(reinterpret_cast<float4*>(O_cooVal + eid + 8)) = *(reinterpret_cast<float4*>(multi + 8));
            *(reinterpret_cast<float4*>(O_cooVal + eid + 12)) = *(reinterpret_cast<float4*>(multi + 12));
        }
    }
    else
    {
        int residue = Size % 16;
        for (int i = 0; i < residue; i++)
        {
            multi[i] = D1_dnVal[S_cooRowInd[eid + i] * D_kcols + cid] * D2_dnVal[S_cooColInd[eid + i] * D_kcols + cid] * S_cooVal[eid + i];
        }
#pragma unroll 4
        for (int i = 0; i < residue; i++)
        {
            atomicAdd(&O_cooVal[eid + i], multi[i]);
        }
    }
}

/*
//a little optimized by share memory
template<typename T>
__global__ void sddmmCOOCacheKernel(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    extern __shared__ T multi[];
    const int eid = blockIdx.x;
    const int cid = threadIdx.x + threadIdx.y * blockDim.x;
    if (cid < D_kcols)
    {
        int offset1, offset2;
        offset1 = S_cooRowInd[eid] * D_kcols;
        offset2 = S_cooColInd[eid] * D_kcols;
        atomicAdd(&multi[eid], D1_dnVal[offset1 + threadIdx.x] * D2_dnVal[offset2 + threadIdx.x] * S_cooVal[eid]);
        __syncthreads();
        int i = blockDim.x / 2;
        while (i != 0)
        {
            if (cacheIdx < i)
                multi[cacheIdx] += multi[cacheIdx + i];
            __syncthreads();
            i /= 2;
        }
        if (cacheIdx == 0)
            dotArr[blockIdx.y] = cache[0];
    }
}
*/


template<typename T>
__global__ void sddmmCSRParKernel(
    int S_mrows, int D_kcols, const unsigned long Size,
    int* S_csrRowPtr, int* S_csrColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_csrVal, T* S_csrVal)
    //imitation of the spmm-kernel-1
    //assign D1_ncols to the blockDim.x so as to dispense with the inner loop
{
    int rid = threadIdx.z + blockIdx.x * blockDim.z;
    int cid = (threadIdx.y << 5) + threadIdx.x;
    if (rid < S_mrows)
    {
        int lb = S_csrRowPtr[rid];
        int hb = S_csrRowPtr[rid + 1];
        int offset1, offset2;
        T acc = 0;
        if (threadIdx.y != blockDim.y - 1)
        {
            for (int ptr = lb; ptr < hb; ptr++)
            {
                offset1 = rid * D_kcols;
                offset2 = S_csrColInd[ptr] * D_kcols;
                acc = D1_dnVal[offset1 + cid] * D2_dnVal[offset2 + cid] * S_csrVal[ptr];
                atomicAdd(&O_csrVal[ptr], acc);
            }
        }
        else
        {
            for (int ptr = lb; ptr < hb; ptr++)
            {
                offset1 = rid * D_kcols;
                offset2 = S_csrColInd[ptr] * D_kcols;
                if (cid < D_kcols) {
                    acc = D1_dnVal[offset1 + cid] * D2_dnVal[offset2 + cid] * S_csrVal[ptr];
                }
                else {
                    acc = 0;
                }
                atomicAdd(&O_csrVal[ptr], acc);
            }
        }
    }
}

template <typename T>
__global__ void sddmmCSRSharedMemory(
    int S_mrows, int D_kcols, const unsigned long Size,
    int* S_csrRowPtr, int* S_csrColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_csrVal, T* S_csrVal)
{
    extern __shared__ T multi[];
    const int rid = blockIdx.x;
    const int cid = (threadIdx.y << 9) + threadIdx.x;
    T* const shmem_ptr = &multi[cid];
    if (rid < S_mrows)
    {
        int lb = S_csrRowPtr[rid];
        int hb = S_csrRowPtr[rid + 1];
        int offset1 = rid * D_kcols;
        for (int ptr = lb; ptr < hb; ptr++)
        {
            int offset2 = S_csrColInd[ptr] * D_kcols;
            T localResult = 0;
            if (cid < D_kcols) {
                localResult = D1_dnVal[offset1 + cid] * D2_dnVal[offset2 + cid] * S_csrVal[ptr];
            }
            *shmem_ptr = localResult;
            for(int i = blockDim.x >> 1;i > 16;i >>= 1)
            {
                __syncthreads();
                if(threadIdx.x < i)
                {
                    localResult += *(shmem_ptr + i);
                    *shmem_ptr = localResult;
                }
            }
            for(int i = 16;i > 1;i >>= 1)
            {
                __syncwarp();
                if (threadIdx.x < i)
                {
                    localResult += *(shmem_ptr + i);
                    *shmem_ptr = localResult;
                }
            }
            if (threadIdx.x == 0)
            {
                localResult += *(shmem_ptr + 1);
                atomicAdd(&O_csrVal[ptr], localResult);
            }
        }
    }
}


template <typename T>
__global__ void sddmmCSRReducKernel(
    int S_mrows, int D_kcols, const unsigned long Size,
    int *S_csrRowPtr, int *S_csrColInd, T *D1_dnVal,
    T *D2_dnVal, T *O_csrVal, T *S_csrVal)
{
    int rid = blockIdx.x;
    int cid = (threadIdx.y << 5) + threadIdx.x;
    int lb = S_csrRowPtr[rid];
    int hb = S_csrRowPtr[(rid + 1)];

    int offset;
    T par_sum, a_val, b_val;
    if (threadIdx.y != blockDim.y - 1)
    {
        // fetch B[rid, cid]
        b_val = D1_dnVal[(rid * D_kcols + cid)];

        for (int ptr = lb; ptr < hb; ptr++)
        {
            // partial_sum = B[rid, cid]*C[rid,cid]
            offset = D_kcols * S_csrColInd[ptr] + cid;
            par_sum = __fmul_rn(b_val, D2_dnVal[offset]);
            a_val = S_csrVal[ptr];
            // reduce among warp
            for (int stride = 16; stride > 0; stride >>= 1)
                par_sum += __shfl_xor_sync(0xffffffff, par_sum, stride, 32);
            // now thread_0 holds \sum{cid=threadIdx.y*32}{threadIdx.y*32+32} (B[rid,cid]*C[rid,cid])
            if (threadIdx.x == 0)
            {
                atomicAdd(&O_csrVal[ptr], par_sum * a_val);
            }
        }
    }
    else
    { // threadIdx.y==blockDim.y-1
        if (cid < D_kcols)
        {
            b_val = D1_dnVal[(rid * D_kcols + cid)];
        }
        else
        {
            b_val = 0;
        }
        for (int ptr = lb; ptr < hb; ptr++)
        {
            // partial_sum = B[rid, cid]*C[rid,cid]
            offset = D_kcols * S_csrColInd[ptr] + cid;
            a_val = S_csrVal[ptr];
            if (cid < D_kcols)
            {
                par_sum = b_val * D2_dnVal[offset];
            }
            else
            {
                par_sum = 0.0;
            }
            // reduce among warp
            for (int stride = 16; stride > 0; stride >>= 1)
                par_sum += __shfl_xor_sync(0xffffffff, par_sum, stride, 32);
            // now thread_0 holds \sum{cid=threadIdx.y*32}{threadIdx.y*32+32} (B[rid,cid]*C[rid,cid])
            if (threadIdx.x == 0)
            {
                atomicAdd(&O_csrVal[ptr], par_sum * a_val);
            }
        }
    }
}

#endif // !sddmmKer_H


enum ProcMethod //processing method for sddmm
{
    CSRSimple,
    CSRReadglobal,
    COOWarpShuffle,
    COOReduction,
    COOPar,
    COOSFUPar,
    CSRPar,
    CSRReduction,
    COOILP2CacheCast,
    COOILP4CacheCast,
    COOILP4CacheCastNext,
    COOILP4CacheCastNext4,
    COOILP4ShReduc,
    COOILP8CacheCast,
    COOILP8CacheCastNext,
    COOILP8CacheCastNext4,
    COOILP16CacheCast,
    COOILP2ShReduc,
    COOILP2ShReducEX,
    COOILP4LoopShReduc,
    COOILP8ShReduc,
    CSRCache,
    CSRCachePar
};


template<typename T, ProcMethod Method, int BlockDimZ>
void sddmmWrapper(
    int S_mrows, int D_kcols, const unsigned long eleSize,
    int* S_csrRowPtr, int* S_csrColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_csrVal, T* S_csrVal
) {
    const size_t sizeOfShare = sizeof(T) * QUOCEIL(D_kcols, 512) * 512;
    switch (Method)
    {
    case CSRSimple:
        sddmmCSRSimple<T> <<<dim3(QUOCEIL(S_mrows, 32), 1, 1), dim3(32, 1, 1) >>> (
            S_mrows, D_kcols, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case CSRPar:
        sddmmCSRParKernel<T> <<<dim3(QUOCEIL(S_mrows, BlockDimZ), 1, 1), dim3(32, QUOCEIL(D_kcols, 32), BlockDimZ)>>> (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case CSRReduction:
        sddmmCSRReducKernel<T> <<<dim3(QUOCEIL(S_mrows, BlockDimZ), 1, 1), dim3(32, QUOCEIL(D_kcols, 32), BlockDimZ)>>> (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case CSRCache:
        sddmmCSRSharedMemory<T> << <dim3(S_mrows, 1, 1), dim3(512, QUOCEIL(D_kcols, 512), 1), sizeOfShare>>> (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOPar:
        sddmmCOOCacheCast<T> <<<dim3(eleSize, 1, 1), dim3(D_kcols, 1, 1)>>> (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;
        
    case COOSFUPar:
        sddmmCOOCacheCastSFU<<<dim3(eleSize, 1, 1), dim3(D_kcols, 1, 1)>>> (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOWarpShuffle:
        sddmmCOOShuffleKernel<T> <<<dim3(QUOCEIL(eleSize, 4), 1, 1), dim3(32, QUOCEIL(D_kcols, 32), 1) >>> (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP2CacheCast:
        sddmmCOOILP2CacheCast<T> << <dim3(QUOCEIL(eleSize, 2), 1, 1), dim3(D_kcols, 1, 1) >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP4CacheCast:
        sddmmCOOILP4CacheCast<T> << <dim3(QUOCEIL(eleSize, 4), 1, 1), dim3(D_kcols, 1, 1) >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP4CacheCastNext:
        sddmmCOOILP4CacheCastNext<T> << <dim3(QUOCEIL(eleSize, 4), 1, 1), dim3(QUOCEIL(D_kcols, 2), 1, 1) >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP4CacheCastNext4:
        sddmmCOOILP4CacheCastNext4<T> << <dim3(QUOCEIL(eleSize, 4), 1, 1), dim3(QUOCEIL(D_kcols, 4), 1, 1) >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP8CacheCast:
        sddmmCOOILP8CacheCast<T> << <dim3(QUOCEIL(eleSize, 8), 1, 1), dim3(D_kcols, 1, 1) >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP8CacheCastNext:
        sddmmCOOILP8CacheCastNext<T> << <dim3(QUOCEIL(eleSize, 8), 1, 1), dim3(QUOCEIL(D_kcols, 2), 1, 1) >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP8CacheCastNext4:
        sddmmCOOILP8CacheCastNext4<T> << <dim3(QUOCEIL(eleSize, 8), 1, 1), dim3(QUOCEIL(D_kcols, 4), 1, 1) >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP16CacheCast:
        sddmmCOOILP16CacheCast<T> << <dim3(QUOCEIL(eleSize, 16), 1, 1), dim3(D_kcols, 1, 1) >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOReduction:
        sddmmCOOReductionSimpleKernel<T> <<<dim3(eleSize, 1, 1), dim3(512, QUOCEIL(D_kcols, 512), 1), sizeOfShare>>> (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP2ShReduc:
        sddmmCOOILP2ShReduc<T> << <dim3(QUOCEIL(eleSize, 2), 1, 1), dim3(512, QUOCEIL(D_kcols, 512), 1), sizeOfShare * 2 >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP2ShReducEX:
        sddmmCOOILP2ShReducEX<T> << <dim3(QUOCEIL(eleSize, 2), 1, 1), dim3(512, QUOCEIL(D_kcols, 512), 1), sizeOfShare * 2 >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;
    
    case COOILP4ShReduc:
        sddmmCOOILP4ShReduc<T> << <dim3(QUOCEIL(eleSize, 4), 1, 1), dim3(512, QUOCEIL(D_kcols, 512), 1), sizeOfShare * 4 >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP4LoopShReduc:
        sddmmCOOILP4LoopShReduc<T> << <dim3(QUOCEIL(eleSize, 4), 1, 1), dim3(512, QUOCEIL(D_kcols, 512), 1), sizeOfShare * 4 >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP8ShReduc:
        sddmmCOOILP8ShReduc<T> << <dim3(QUOCEIL(eleSize, 8), 1, 1), dim3(512, QUOCEIL(D_kcols, 512), 1), sizeOfShare * 8 >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;
    }
}

template<typename T>
__global__ void Mutiply(unsigned long eleSize, T* O_csrVal, T* S_csrVal)
{
    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = tidy * blockDim.x * gridDim.x + tidx;
    O_csrVal[tid] *= S_csrVal[tid];
}

//this is the C method of read COO and transfer it to CSR
//deprecated when you want to read the data directly from the CSR generated

//transfer the COO to CSR
void COO_to_CSR(std::vector<int>& row_CSR, std::vector<int> row_COO,
    unsigned long Size, int matrixRow)
{
    row_CSR.push_back(0);
    if (row_COO[0] != 0)
    {
        for (int j = 0; j < row_COO[0]; j++)
            row_CSR.push_back(0);
    }
    for (int i = 0; i < (Size - 1); i++)
    {
        for (int j = 0; j < row_COO[i + 1] - row_COO[i]; j++)
        {
            row_CSR.push_back(i + 1);
        }
    }
    for (int j = 0; j < matrixRow + 1 - row_COO.back(); j++)
    {
        row_CSR.push_back(static_cast<int>(Size));
    }
}

//read for arbitary COO (generated^_^)
template <typename T>
unsigned long readCOO(const char* file, std::vector<int>& row_indices,
    std::vector<int>& col_indices, std::vector<T>& values,
    int& S_mrows, int& S_ncols, int& D_kcols, unsigned long Size)
{
    int col_element, row_element;
    float value;
    std::ifstream fm(file, std::ios::in);
    if (!fm)
        std::cerr << "cannot open the file!\n";
    else
    {
        fm >> S_mrows >> S_ncols >> D_kcols >> Size;
        do
        {
            fm >> row_element;
            fm >> col_element;
            fm >> value;
            if (fm.fail())
                break;
            col_indices.push_back(col_element);
            row_indices.push_back(row_element);
            values.push_back(value);
        } while (!fm.eof());
    }
    fm.close();
    return col_indices.size();
}

template <typename T>
void readCSR(const char* file,
    std::vector<int> &S_rowPtr, std::vector<int> &S_colInd, std::vector<T> &S_csrVal,
    int& S_mrows, int& S_ncols, int& D_kcols, unsigned long &eleSize)
{
    std::ifstream fr(file, std::ios::in);
    if (!fr)
        std::cerr << "cannot open the file" << std::endl;
    int indptr_Size, indices_Size, data_Size;
    fr >> S_mrows >> S_ncols >> D_kcols >> eleSize;
    fr >> indptr_Size >> indices_Size >> data_Size;
    int row, col;
    T val;
    for (int i = 0; i < indptr_Size; i++)
    {
        fr >> row;
        S_rowPtr.push_back(row);
    }
    for (int i = 0; i < indices_Size; i++)
    {
        fr >> col;
        S_colInd.push_back(col);
    }
    for (int i = 0; i < indices_Size; i++)
    {
        fr >> val;
        S_csrVal.push_back(val);
    }
    fr.close();
}

template <typename T>
void readVecMat(const char* file, T* dense)
{
    std::ifstream fr(file, std::ios::in);
    if (!fr)
        std::cerr << "cannot open the file!" << std::endl;
    int count = 0;
    do
    {
        fr >> dense[count];
        count++;
    } while (!fr.eof());
    fr.close();
}

struct timeStruct
{
    float time;
    char name[30];
    timeStruct(float a, const char* b)
    {
        time = a;
        strcpy(name, b);
    };
};

bool compare(const timeStruct &a, const timeStruct &b)
{
    return (a.time < b.time);
}