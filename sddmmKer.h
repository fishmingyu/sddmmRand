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
__global__ void sddmmCOOSimpleKernel(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    const int eid = blockIdx.x;
    const int cid = threadIdx.x + threadIdx.y * blockDim.x; //the id for column
    int offset1 = S_cooRowInd[eid] * D_kcols + cid;
    int offset2 = S_cooColInd[eid] * D_kcols + cid ;
    T multi = D1_dnVal[offset1] * D2_dnVal[offset2] * S_cooVal[eid];
    atomicAdd(&O_cooVal[eid], multi);
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
    for (int stride = 16; stride > 0; stride /= 2)
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
__global__ void sddmmCOOILP2Cache(int S_mrows, int D_kcols, const unsigned long Size,
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
__global__ void sddmmCOOILP4Cache(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    extern __shared__ T multi[];
    int multiOffset = QUOCEIL(D_kcols, 512) * 512;
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

    T* const smem_ptr[4] = { &multi[cid], &multi[cid + multiOffset], &multi[cid + 2 * multiOffset], &multi[cid + 3 * multiOffset] };
    *smem_ptr[0] = localResult[0];
    *smem_ptr[1] = localResult[1];
    *smem_ptr[2] = localResult[2];
    *smem_ptr[3] = localResult[3];

    __syncthreads();
    if (threadIdx.x < 256){
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
    if (threadIdx.x < 128){
        localResult[0] += *(smem_ptr[0] + 128);
        localResult[1] += *(smem_ptr[1] + 128);
        localResult[2] += *(smem_ptr[2] + 128);
        localResult[3] += *(smem_ptr[3] + 128);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
        *smem_ptr[2] = localResult[2];
        *smem_ptr[3] = localResult[3];
    }

    __syncthreads();
    if (threadIdx.x < 64){
        localResult[0] += *(smem_ptr[0] + 64);
        localResult[1] += *(smem_ptr[1] + 64);
        localResult[2] += *(smem_ptr[2] + 64);
        localResult[3] += *(smem_ptr[3] + 64);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
        *smem_ptr[2] = localResult[2];
        *smem_ptr[3] = localResult[3];
    }

    __syncthreads();
    if (threadIdx.x < 32){
        localResult[0] += *(smem_ptr[0] + 32);
        localResult[1] += *(smem_ptr[1] + 32);
        localResult[2] += *(smem_ptr[2] + 32);
        localResult[3] += *(smem_ptr[3] + 32);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
        *smem_ptr[2] = localResult[2];
        *smem_ptr[3] = localResult[3];
    }

    __syncthreads();
    if (threadIdx.x < 16){
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
    if(threadIdx.x < 8)
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
    __syncwarp();
    if(threadIdx.x < 4)
    {
        localResult[0] += *(smem_ptr[0] + 4);
        localResult[1] += *(smem_ptr[1] + 4);
        localResult[2] += *(smem_ptr[2] + 4);
        localResult[3] += *(smem_ptr[3] + 4);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
        *smem_ptr[2] = localResult[2];
        *smem_ptr[3] = localResult[3];
    }
    __syncwarp();
    if(threadIdx.x < 2)
    {
        localResult[0] += *(smem_ptr[0] + 2);
        localResult[1] += *(smem_ptr[1] + 2);
        localResult[2] += *(smem_ptr[2] + 2);
        localResult[3] += *(smem_ptr[3] + 2);
        *smem_ptr[0] = localResult[0];
        *smem_ptr[1] = localResult[1];
        *smem_ptr[2] = localResult[2];
        *smem_ptr[3] = localResult[3];
    }
    if (threadIdx.x == 0) {
        if(eid != Size - 3)
        {
            localResult[0] += *(smem_ptr[0] + 1);
            localResult[1] += *(smem_ptr[1] + 1);
            localResult[2] += *(smem_ptr[2] + 1);
            localResult[3] += *(smem_ptr[3] + 1);
            
            atomicAdd(&O_cooVal[eid], localResult[0]);
            atomicAdd(&O_cooVal[eid + 1], localResult[1]);
            atomicAdd(&O_cooVal[eid + 2], localResult[2]);
            atomicAdd(&O_cooVal[eid + 3], localResult[3]);
        }
        else
        {
            for(int i = 0;i < Size % 4; i++){
                localResult[i] += *(smem_ptr[i] + 1);
                atomicAdd(&O_cooVal[eid + i], localResult[i]);
            }
        }
    }
}


template<typename T>
__global__ void sddmmCOOILP4LoopCache(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    extern __shared__ T multi[];
    int multiOffset = QUOCEIL(D_kcols, 512) * 512;
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

    T* const smem_ptr[4] = { &multi[cid], &multi[cid + multiOffset], &multi[cid + 2 * multiOffset], &multi[cid + 3 * multiOffset] };
    *smem_ptr[0] = localResult[0];
    *smem_ptr[1] = localResult[1];
    *smem_ptr[2] = localResult[2];
    *smem_ptr[3] = localResult[3];

    for(int i = blockDim.x >> 1;i > 16;i >>= 1)
    {
        __syncthreads();
        if(threadIdx.x < i){
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
    for(int i = 16;i > 1;i >>= 1)
    {
        __syncwarp();
        if(threadIdx.x < i){
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
            localResult[0] += *(smem_ptr[0] + 1);
            localResult[1] += *(smem_ptr[1] + 1);
            localResult[2] += *(smem_ptr[2] + 1);
            localResult[3] += *(smem_ptr[3] + 1);
            
            atomicAdd(&O_cooVal[eid], localResult[0]);
            atomicAdd(&O_cooVal[eid + 1], localResult[1]);
            atomicAdd(&O_cooVal[eid + 2], localResult[2]);
            atomicAdd(&O_cooVal[eid + 3], localResult[3]);
        }
        else
        {
            for(int i = 0;i < Size % 4; i++){
                localResult[i] += *(smem_ptr[i] + 1);
                atomicAdd(&O_cooVal[eid + i], localResult[i]);
            }
        }
    }
}

template<typename T>
__global__ void sddmmCOOILP8Cache(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    extern __shared__ T multi[];
    int multiOffset = QUOCEIL(D_kcols, 512) * 512;
    int eid = (blockIdx.x) << 3;
    int cid = (threadIdx.y << 9) + threadIdx.x;
    int offset1[8], offset2[8];
    T localResult[8] = { 0, 0, 0, 0 ,0 ,0, 0, 0};
    //Compute the vector dot production using ILP4
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
__global__ void sddmmCOOILP2Kernel(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    const int eid = (blockIdx.x + blockIdx.y * gridDim.x) << 1;
    const int cid = threadIdx.x + threadIdx.y * blockDim.x;
    int offset1[2], offset2[2];
    T multi[2] = {0, 0};
    if (eid != Size - 1)
    {
        offset1[0] = S_cooRowInd[eid] * D_kcols;
        offset1[1] = S_cooRowInd[eid + 1] * D_kcols;
        offset2[0] = S_cooColInd[eid] * D_kcols;
        offset2[1] = S_cooColInd[eid + 1] * D_kcols;
        multi[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
        multi[1] = D1_dnVal[offset1[1] + cid] * D2_dnVal[offset2[1] + cid] * S_cooVal[eid + 1];
        atomicAdd(&O_cooVal[eid], multi[0]);
        atomicAdd(&O_cooVal[eid + 1], multi[1]);
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
__global__ void sddmmCOOILP4Kernel(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    const int eid = (blockIdx.x + blockIdx.y * gridDim.x) << 2;
    const int cid = threadIdx.x + threadIdx.y * blockDim.x;
    int offset1[4], offset2[4];
    T multi[4] = { 0, 0, 0, 0 };
    if (eid < Size - 3)
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
__global__ void sddmmCOOILP4exKernel(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    const int eid = (blockIdx.x + blockIdx.y * gridDim.x) << 2;
    const int cid = threadIdx.x + threadIdx.y * blockDim.x;
    int offset1[4], offset2[4];
    T multi[4] = { 0, 0, 0, 0 };
    if (eid < Size - 3)
    {
        offset1[0] = *(S_cooRowInd + eid) * D_kcols;
        offset1[1] = *(S_cooRowInd + eid + 1) * D_kcols;
        offset1[2] = *(S_cooRowInd + eid + 2) * D_kcols;
        offset1[3] = *(S_cooRowInd + eid + 3) * D_kcols;
        offset2[0] = *(S_cooColInd + eid) * D_kcols;
        offset2[1] = *(S_cooColInd + eid + 1) * D_kcols;
        offset2[2] = *(S_cooColInd + eid + 2) * D_kcols;
        offset2[3] = *(S_cooColInd + eid + 3) * D_kcols;
        multi[0] = *(D1_dnVal + offset1[0] + cid) * *(D2_dnVal + offset2[0] + cid) * *(S_cooVal + eid);
        multi[1] = *(D1_dnVal + offset1[1] + cid) * *(D2_dnVal + offset2[1] + cid) * *(S_cooVal + eid + 1);
        multi[2] = *(D1_dnVal + offset1[2] + cid) * *(D2_dnVal + offset2[2] + cid) * *(S_cooVal + eid + 2);
        multi[3] = *(D1_dnVal + offset1[3] + cid) * *(D2_dnVal + offset2[3] + cid) * *(S_cooVal + eid + 3);
        atomicAdd(&O_cooVal[eid], multi[0]);
        atomicAdd(&O_cooVal[eid + 1], multi[1]);
        atomicAdd(&O_cooVal[eid + 2], multi[2]);
        atomicAdd(&O_cooVal[eid + 3], multi[3]);
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
__global__ void sddmmCOOILP4UnrollKernel(int S_mrows, int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    const int eid = 4 * (blockIdx.x + blockIdx.y * gridDim.x);
    const int cid = threadIdx.x + threadIdx.y * blockDim.x;
    int offset1[4], offset2[4];
    T multi[4] = { 0, 0, 0, 0 };
    if (eid < Size - 3)
    {
#pragma unroll 4
        for (int i = 0; i < 4; i++)
        {
            offset1[i] = S_cooRowInd[eid + i] * D_kcols;
        }
#pragma unroll 4
        for (int i = 0; i < 4; i++)
        {
            offset2[i] = S_cooColInd[eid + i] * D_kcols;
        }
#pragma unroll 4
        for (int i = 0; i < 4; i++)
        {
            multi[i] = D1_dnVal[offset1[i] + cid] * D2_dnVal[offset2[i] + cid] * S_cooVal[eid + i];
        }
#pragma unroll 4
        for (int i = 0; i < 4; i++)
        {
            atomicAdd(&O_cooVal[eid + i], multi[i]);
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
__global__ void sddmmCOOILP8Kernel(int S_mrows, const int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    const int eid = (blockIdx.x + blockIdx.y * gridDim.x) << 3;
    const int cid = threadIdx.x + threadIdx.y * blockDim.x;
    int offset1[8], offset2[8];
    T multi[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    if (eid < Size - 7)
    {
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

        multi[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
        multi[1] = D1_dnVal[offset1[1] + cid] * D2_dnVal[offset2[1] + cid] * S_cooVal[eid + 1];
        multi[2] = D1_dnVal[offset1[2] + cid] * D2_dnVal[offset2[2] + cid] * S_cooVal[eid + 2];
        multi[3] = D1_dnVal[offset1[3] + cid] * D2_dnVal[offset2[3] + cid] * S_cooVal[eid + 3];
        multi[4] = D1_dnVal[offset1[4] + cid] * D2_dnVal[offset2[4] + cid] * S_cooVal[eid + 4];
        multi[5] = D1_dnVal[offset1[5] + cid] * D2_dnVal[offset2[5] + cid] * S_cooVal[eid + 5];
        multi[6] = D1_dnVal[offset1[6] + cid] * D2_dnVal[offset2[6] + cid] * S_cooVal[eid + 6];
        multi[7] = D1_dnVal[offset1[7] + cid] * D2_dnVal[offset2[7] + cid] * S_cooVal[eid + 7];

        atomicAdd(&O_cooVal[eid], multi[0]);
        atomicAdd(&O_cooVal[eid + 1], multi[1]);
        atomicAdd(&O_cooVal[eid + 2], multi[2]);
        atomicAdd(&O_cooVal[eid + 3], multi[3]);
        atomicAdd(&O_cooVal[eid + 4], multi[4]);
        atomicAdd(&O_cooVal[eid + 5], multi[5]);
        atomicAdd(&O_cooVal[eid + 6], multi[6]);
        atomicAdd(&O_cooVal[eid + 7], multi[7]);
    }
    else
    {
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
__global__ void sddmmCOOILP16Kernel(int S_mrows, const int D_kcols, const unsigned long Size,
    int* S_cooRowInd, int* S_cooColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_cooVal, T* S_cooVal)
{
    const int eid = (blockIdx.x + blockIdx.y * gridDim.x) << 4;
    const int cid = threadIdx.x + threadIdx.y * blockDim.x;
    int offset1[16], offset2[16];
    T multi[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    if (eid < Size - 15)
    {
        offset1[0] = S_cooRowInd[eid] * D_kcols;
        offset1[1] = S_cooRowInd[eid + 1] * D_kcols;
        offset1[2] = S_cooRowInd[eid + 2] * D_kcols;
        offset1[3] = S_cooRowInd[eid + 3] * D_kcols;
        offset1[4] = S_cooRowInd[eid + 4] * D_kcols;
        offset1[5] = S_cooRowInd[eid + 5] * D_kcols;
        offset1[6] = S_cooRowInd[eid + 6] * D_kcols;
        offset1[7] = S_cooRowInd[eid + 7] * D_kcols;
        offset1[8] = S_cooRowInd[eid + 8] * D_kcols;
        offset1[9] = S_cooRowInd[eid + 9] * D_kcols;
        offset1[10] = S_cooRowInd[eid + 10] * D_kcols;
        offset1[11] = S_cooRowInd[eid + 11] * D_kcols;
        offset1[12] = S_cooRowInd[eid + 12] * D_kcols;
        offset1[13] = S_cooRowInd[eid + 13] * D_kcols;
        offset1[14] = S_cooRowInd[eid + 14] * D_kcols;
        offset1[15] = S_cooRowInd[eid + 15] * D_kcols;

        offset2[0] = S_cooColInd[eid] * D_kcols;
        offset2[1] = S_cooColInd[eid + 1] * D_kcols;
        offset2[2] = S_cooColInd[eid + 2] * D_kcols;
        offset2[3] = S_cooColInd[eid + 3] * D_kcols;
        offset2[4] = S_cooColInd[eid + 4] * D_kcols;
        offset2[5] = S_cooColInd[eid + 5] * D_kcols;
        offset2[6] = S_cooColInd[eid + 6] * D_kcols;
        offset2[7] = S_cooColInd[eid + 7] * D_kcols;
        offset2[8] = S_cooColInd[eid + 8] * D_kcols;
        offset2[9] = S_cooColInd[eid + 9] * D_kcols;
        offset2[10] = S_cooColInd[eid + 10] * D_kcols;
        offset2[11] = S_cooColInd[eid + 11] * D_kcols;
        offset2[12] = S_cooColInd[eid + 12] * D_kcols;
        offset2[13] = S_cooColInd[eid + 13] * D_kcols;
        offset2[14] = S_cooColInd[eid + 14] * D_kcols;
        offset2[15] = S_cooColInd[eid + 15] * D_kcols;

        multi[0] = D1_dnVal[offset1[0] + cid] * D2_dnVal[offset2[0] + cid] * S_cooVal[eid];
        multi[1] = D1_dnVal[offset1[1] + cid] * D2_dnVal[offset2[1] + cid] * S_cooVal[eid + 1];
        multi[2] = D1_dnVal[offset1[2] + cid] * D2_dnVal[offset2[2] + cid] * S_cooVal[eid + 2];
        multi[3] = D1_dnVal[offset1[3] + cid] * D2_dnVal[offset2[3] + cid] * S_cooVal[eid + 3];
        multi[4] = D1_dnVal[offset1[4] + cid] * D2_dnVal[offset2[4] + cid] * S_cooVal[eid + 4];
        multi[5] = D1_dnVal[offset1[5] + cid] * D2_dnVal[offset2[5] + cid] * S_cooVal[eid + 5];
        multi[6] = D1_dnVal[offset1[6] + cid] * D2_dnVal[offset2[6] + cid] * S_cooVal[eid + 6];
        multi[7] = D1_dnVal[offset1[7] + cid] * D2_dnVal[offset2[7] + cid] * S_cooVal[eid + 7];
        multi[8] = D1_dnVal[offset1[8] + cid] * D2_dnVal[offset2[8] + cid] * S_cooVal[eid + 8];
        multi[9] = D1_dnVal[offset1[9] + cid] * D2_dnVal[offset2[9] + cid] * S_cooVal[eid + 9];
        multi[10] = D1_dnVal[offset1[10] + cid] * D2_dnVal[offset2[10] + cid] * S_cooVal[eid + 10];
        multi[11] = D1_dnVal[offset1[11] + cid] * D2_dnVal[offset2[11] + cid] * S_cooVal[eid + 11];
        multi[12] = D1_dnVal[offset1[12] + cid] * D2_dnVal[offset2[12] + cid] * S_cooVal[eid + 12];
        multi[13] = D1_dnVal[offset1[13] + cid] * D2_dnVal[offset2[13] + cid] * S_cooVal[eid + 13];
        multi[14] = D1_dnVal[offset1[14] + cid] * D2_dnVal[offset2[14] + cid] * S_cooVal[eid + 14];
        multi[15] = D1_dnVal[offset1[15] + cid] * D2_dnVal[offset2[15] + cid] * S_cooVal[eid + 15];

        atomicAdd(&O_cooVal[eid], multi[0]);
        atomicAdd(&O_cooVal[eid + 1], multi[1]);
        atomicAdd(&O_cooVal[eid + 2], multi[2]);
        atomicAdd(&O_cooVal[eid + 3], multi[3]);
        atomicAdd(&O_cooVal[eid + 4], multi[4]);
        atomicAdd(&O_cooVal[eid + 5], multi[5]);
        atomicAdd(&O_cooVal[eid + 6], multi[6]);
        atomicAdd(&O_cooVal[eid + 7], multi[7]);
        atomicAdd(&O_cooVal[eid + 8], multi[8]);
        atomicAdd(&O_cooVal[eid + 9], multi[9]);
        atomicAdd(&O_cooVal[eid + 10], multi[10]);
        atomicAdd(&O_cooVal[eid + 11], multi[11]);
        atomicAdd(&O_cooVal[eid + 12], multi[12]);
        atomicAdd(&O_cooVal[eid + 13], multi[13]);
        atomicAdd(&O_cooVal[eid + 14], multi[14]);
        atomicAdd(&O_cooVal[eid + 15], multi[15]);
    }
    else
    {
        int residue = Size % 16;
#pragma unroll 4
        for (int i = 0; i < residue; i++)
        {
            offset1[i] = S_cooRowInd[eid + i] * D_kcols;
        }
#pragma unroll 4
        for (int i = 0; i < residue; i++)
        {
            offset2[i] = S_cooColInd[eid + i] * D_kcols;
        }
#pragma unroll 4
        for (int i = 0; i < residue; i++)
        {
            multi[i] = D1_dnVal[offset1[i] + cid] * D2_dnVal[offset2[i] + cid] * S_cooVal[eid + i];
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

//experiment change the logic sequence
template<typename T>
__global__ void sddmmCSRParexKernel(
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
        int offset1 = rid * D_kcols;
        for (int ptr = lb; ptr < hb; ptr++)
        {
            int offset2 = S_csrColInd[ptr] * D_kcols;
            T acc = 0;
            if (cid < D_kcols) {
                acc = D1_dnVal[offset1 + cid] * D2_dnVal[offset2 + cid] * S_csrVal[ptr];
            }
            atomicAdd(&O_csrVal[ptr], acc);
        }
    }
}

template <typename T>
__global__ void sddmmCSRRec2(
    int S_mrows, int D_kcols, const unsigned long Size,
    int* S_csrRowPtr, int* S_csrColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_csrVal, T* S_csrVal)
{
    extern __shared__ T DCache[];
    int rid = blockIdx.x;
    int cid = (threadIdx.y << 5) + threadIdx.x;
    int lb = S_csrRowPtr[rid];
    int hb = S_csrRowPtr[(rid + 1)];
    int offset;
    T par_sum, a_val;
    if(cid < D_kcols)
        DCache[cid] = D1_dnVal[(rid * D_kcols + cid)];
    __syncthreads();
    if (threadIdx.y != blockDim.y - 1) {
        for (int ptr = lb; ptr < hb; ptr++) {
            offset = D_kcols * S_csrColInd[ptr] + cid;
            par_sum = DCache[cid] * D2_dnVal[offset];
            a_val = S_csrVal[ptr];
            for (int stride = 16; stride > 0; stride /= 2)
                par_sum += __shfl_xor_sync(0xffffffff, par_sum, stride, 32);
            if (threadIdx.x == 0) {
                atomicAdd(&O_csrVal[ptr], par_sum * a_val);
            }
        }
    }
    else { // threadIdx.y==blockDim.y-1
        for (int ptr = lb; ptr < hb; ptr++) {
            offset = D_kcols * S_csrColInd[ptr] + cid;
            a_val = S_csrVal[ptr];
            if (cid < D_kcols) {
                par_sum = DCache[cid] * D2_dnVal[offset];
            }
            else {
                par_sum = 0.0;
            }// reduce among warp
            for (int stride = 16; stride > 0; stride /= 2)
                par_sum += __shfl_xor_sync(0xffffffff, par_sum, stride, 32);
            if (threadIdx.x == 0) {
                atomicAdd(&O_csrVal[ptr], par_sum * a_val);
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
__global__ void sddmmCSRWarpRec(
    int S_mrows, int D_kcols, const unsigned long Size,
    int* S_csrRowPtr, int* S_csrColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_csrVal, T* S_csrVal)
{
    extern __shared__ T CSRCache[];
    const int rid = blockIdx.x;
    const int cid = (threadIdx.y << 5) + threadIdx.x;
    const int lb = S_csrRowPtr[rid];
    const int hb = S_csrRowPtr[rid + 1];
    int offset2;
    T D1Reg;
    if(cid < D_kcols)
        D1Reg = D1_dnVal[rid * D_kcols + cid]; 
    else
        D1Reg = 0.0;
    if(cid < (hb - lb + 1)) //the amount of elements in one row
    {
        CSRCache[cid] = S_csrVal[cid + lb];
    }
    __syncthreads();
    if (threadIdx.y != blockDim.y - 1){
        T acc; 
        for (int ptr = lb; ptr < hb; ptr++)
        {
            int offsetCache = ptr - lb;
            offset2 = S_csrColInd[ptr] * D_kcols + cid;
            acc = D1Reg * D2_dnVal[offset2];
            for (int stride = 16; stride > 0; stride /= 2)
                acc += __shfl_xor_sync(0xffffffff, acc, stride, 32);
            if (threadIdx.x == 0) {
                atomicAdd(&O_csrVal[ptr], acc * CSRCache[offsetCache]);
            }
        }
    }
    else { // threadIdx.y==blockDim.y-1
        T acc;
        for (int ptr = lb; ptr < hb; ptr++) {
            int offsetCache = ptr - lb;
            offset2 = S_csrColInd[ptr] * D_kcols + cid;
            if (cid < D_kcols) {
                acc = D1Reg * D2_dnVal[offset2];
            }
            else
            {
                acc = 0.0;
            }
            // reduce among warp
            for (int stride = 16; stride > 0; stride /= 2)
                acc += __shfl_xor_sync(0xffffffff, acc, stride, 32);
            if (threadIdx.x == 0) {
                atomicAdd(&O_csrVal[ptr], acc * CSRCache[offsetCache]);
            }
        }
    }
}

template<typename T>
__global__ void sddmmCSRReducKernel(
    int S_mrows, int D_kcols, const unsigned long Size,
    int* S_csrRowPtr, int* S_csrColInd, T* D1_dnVal,
    T* D2_dnVal, T* O_csrVal, T* S_csrVal)
{
    int rid = threadIdx.z + blockIdx.x * blockDim.z;
    if (rid < S_mrows) {
        int cid = (threadIdx.y << 5) + threadIdx.x;
        int lb = S_csrRowPtr[rid];
        int hb = S_csrRowPtr[(rid + 1)];

        int offset;
        T par_sum, a_val, b_val;
        if (threadIdx.y != blockDim.y - 1) {
            // fetch B[rid, cid]
            b_val = D1_dnVal[(rid * D_kcols + cid)];

            for (int ptr = lb; ptr < hb; ptr++) {
                // partial_sum = B[rid, cid]*C[rid,cid]
                offset = D_kcols * S_csrColInd[ptr] + cid;
                par_sum = b_val * D2_dnVal[offset];
                a_val = S_csrVal[ptr];
                // reduce among warp
                for (int stride = 16; stride > 0; stride /= 2)
                    par_sum += __shfl_xor_sync(0xffffffff, par_sum, stride, 32);
                // now thread_0 holds \sum{cid=threadIdx.y*32}{threadIdx.y*32+32} (B[rid,cid]*C[rid,cid])
                if (threadIdx.x == 0) {
                    atomicAdd(&O_csrVal[ptr], par_sum * a_val);
                }
            }
        }
        else { // threadIdx.y==blockDim.y-1
            if (cid < D_kcols) {
                b_val = D1_dnVal[(rid * D_kcols + cid)];
            }
            else {
                b_val = 0;
            }
            for (int ptr = lb; ptr < hb; ptr++) {
                // partial_sum = B[rid, cid]*C[rid,cid]
                offset = D_kcols * S_csrColInd[ptr] + cid;
                a_val = S_csrVal[ptr];
                if (cid < D_kcols) {
                    par_sum = b_val * D2_dnVal[offset];
                }
                else {
                    par_sum = 0.0;
                }
                // reduce among warp
                for (int stride = 16; stride > 0; stride /= 2)
                    par_sum += __shfl_xor_sync(0xffffffff, par_sum, stride, 32);
                // now thread_0 holds \sum{cid=threadIdx.y*32}{threadIdx.y*32+32} (B[rid,cid]*C[rid,cid])
                if (threadIdx.x == 0) {
                    atomicAdd(&O_csrVal[ptr], par_sum * a_val);
                }
            }
        }
    }
}



#endif // !sddmmKer_H


enum ProcMethod //processing method for sddmm
{
    CSRSimple,
    CSRReadglobal,
    COOSimple,
    COOWarpShuffle,
    COOReduction,
    COOPar,
    CSRPar,
    CSRParex,
    CSRWarpRec,
    CSRWarpRec2,
    CSRReduction,
    COOILP2,
    COOILP4,
    COOILP4ex,
    COOILP4Unroll,
    COOILP8,
    COOILP16,
    COOILP2Cache,
    COOILP4Cache,
    COOILP4LoopCache,
    COOILP8Cache,
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
    const size_t sizeOfShareWarp = sizeof(T) * D_kcols * 2;
    const size_t sizeOfShareD = sizeof(T) * D_kcols;
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

    case CSRParex:
        sddmmCSRParexKernel<T> << <dim3(QUOCEIL(S_mrows, BlockDimZ), 1, 1), dim3(32, QUOCEIL(D_kcols, 32), BlockDimZ) >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOSimple:
        sddmmCOOSimpleKernel<T> <<<dim3(eleSize, 1, 1), dim3(D_kcols, 1, 1) >>> (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOWarpShuffle:
        sddmmCOOShuffleKernel<T> <<<dim3(QUOCEIL(eleSize, 4), 1, 1), dim3(32, QUOCEIL(D_kcols, 32), 1) >>> (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP2:
        sddmmCOOILP2Kernel<T> << <dim3(QUOCEIL(eleSize, 2), 1, 1), dim3(D_kcols, 1, 1) >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP4:
        sddmmCOOILP4Kernel<T> << <dim3(QUOCEIL(eleSize, 4), 1, 1), dim3(D_kcols, 1, 1) >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP4ex:
        sddmmCOOILP4exKernel<T> << <dim3(QUOCEIL(eleSize, 4), 1, 1), dim3(D_kcols, 1, 1) >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP4Unroll:
        sddmmCOOILP4UnrollKernel<T> << <dim3(QUOCEIL(eleSize, 4), 1, 1), dim3(D_kcols, 1, 1) >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP8:
        sddmmCOOILP8Kernel<T> << <dim3(QUOCEIL(eleSize, 8), 1, 1), dim3(D_kcols, 1, 1) >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP16:
        sddmmCOOILP16Kernel<T> << <dim3(QUOCEIL(eleSize, 16), 1, 1), dim3(D_kcols, 1, 1) >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOReduction:
        sddmmCOOReductionSimpleKernel<T> <<<dim3(eleSize, 1, 1), dim3(512, QUOCEIL(D_kcols, 512), 1), sizeOfShare>>> (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP2Cache:
        sddmmCOOILP2Cache<T> << <dim3(QUOCEIL(eleSize, 2), 1, 1), dim3(512, QUOCEIL(D_kcols, 512), 1), sizeOfShare * 2 >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;
    
    case COOILP4Cache:
        sddmmCOOILP4Cache<T> << <dim3(QUOCEIL(eleSize, 4), 1, 1), dim3(512, QUOCEIL(D_kcols, 512), 1), sizeOfShare * 4 >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP4LoopCache:
        sddmmCOOILP4LoopCache<T> << <dim3(QUOCEIL(eleSize, 4), 1, 1), dim3(512, QUOCEIL(D_kcols, 512), 1), sizeOfShare * 4 >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case COOILP8Cache:
        sddmmCOOILP8Cache<T> << <dim3(QUOCEIL(eleSize, 8), 1, 1), dim3(512, QUOCEIL(D_kcols, 512), 1), sizeOfShare * 8 >> > (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case CSRWarpRec:
        sddmmCSRWarpRec<T> <<<dim3(QUOCEIL(S_mrows, BlockDimZ), 1, 1), dim3(32, QUOCEIL(D_kcols, 32), BlockDimZ), sizeOfShareWarp >>> (
            S_mrows, D_kcols, eleSize, S_csrRowPtr, S_csrColInd, D1_dnVal, D2_dnVal, O_csrVal, S_csrVal);
        break;

    case CSRWarpRec2:
        sddmmCSRRec2<T> << <dim3(QUOCEIL(S_mrows, BlockDimZ), 1, 1), dim3(32, QUOCEIL(D_kcols, 32), BlockDimZ), sizeOfShareD>> > (
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
