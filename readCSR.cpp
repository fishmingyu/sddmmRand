#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>


/*if you want to process the data with C, this demo is provide some simple examples*/
std::string file1 = "sparseCOO.txt";
std::string file2 = "indptr.txt";
std::vector<int> row_COO, col_IND, row_CSR, py_rowCSR;
std::vector<float> values;

int* S_csrRowPtr = nullptr, * S_csrColInd = nullptr;
float* S_csrVal = nullptr;
int S_mrows, S_ncols, D_kcols;
unsigned long Size;

template <typename T>
void readCOO(std::string file, std::vector<int> &row_indices,
             std::vector<int> &col_indices, std::vector<T> &values,
                      int &S_mrows, int &S_ncols, int &D_kcols, unsigned long &Size);
void COO_to_CSR(std::vector<int> &row_CSR,
                std::vector<int> row_COO, unsigned long Size, int matrixRow);
void readFromCOO();
void readFromCSR(std::string file);

int main()
{
    readFromCOO();
    readFromCSR(file2);
    bool fal = true;
    //test for uniformity from two methods
    for(int i = 0;i < py_rowCSR.size();i++)
    {
        if(py_rowCSR[i] != S_csrRowPtr[i])
        {
            std::cout << "error in CSR"<< i <<"!\n";
            fal = false;
        }
    }
    if(fal)
        std::cout << "verified successfully" << std::endl;
    free(S_csrVal);
    free(S_csrRowPtr);
    free(S_csrColInd);
    return 0;
}


//read from the file and store
template <typename T>
void readCOO(std::string file, std::vector<int> &row_indices,
             std::vector<int> &col_indices, std::vector<T> &values,
                      int &S_mrows, int &S_ncols, int &D_kcols, unsigned long &eleSize)
{
    int col_element, row_element;
    float value;
    std::ifstream fm(file, std::ios::in);
    if(!fm)
        std::cerr << "cannot open the file!\n";
    else
    {
        fm >> S_mrows >> S_ncols >> D_kcols >> eleSize;
        do
        {
            fm >> col_element;
            fm >> row_element;
            fm >> value;
            if(fm.fail())
                break;
            col_indices.push_back(col_element);
            row_indices.push_back(row_element);
            values.push_back(value);
        }while(!fm.eof());
    }
}

//transfer the COO to CSR
void COO_to_CSR(std::vector<int> &row_CSR, std::vector<int> row_COO,
                unsigned long Size, int S_mrows)
{
    row_CSR.push_back(0);
    if(row_COO[0] != 0)
    {
        for(int j = 0;j < row_COO[0];j++)
            row_CSR.push_back(0);
    }
    for(int i = 0;i < (Size - 1);i++)
    {
        for(int j = 0;j < row_COO[i + 1] - row_COO[i];j++)
        {
            row_CSR.push_back(i + 1);
        }
    }
    for(int j = 0;j < S_mrows + 1- row_COO.back();j++)
    {
        row_CSR.push_back(static_cast<int>(Size));
    }
}

void readFromCOO()
{
    readCOO<float>(file1, row_COO, col_IND, values, S_mrows, S_ncols, D_kcols, Size);
    S_csrColInd = (int*)malloc(Size * sizeof(int));
    S_csrVal = (float*)malloc(Size * sizeof(float));
    std::sort(row_COO.begin(), row_COO.end());
    COO_to_CSR(row_CSR, row_COO, Size, S_mrows);
    S_csrRowPtr = (int*)malloc(row_CSR.size() * sizeof(int));
    std::copy(row_CSR.begin(), row_CSR.end(), S_csrRowPtr);
    std::copy(col_IND.begin(), col_IND.end(), S_csrColInd);
    std::copy(values.begin(), values.end(), S_csrVal);
}

void readFromCSR(std::string file)
{
    std::ifstream fm(file, std::ios::in);
    if(!fm)
        std::cerr << "cannot open the file!\n";
    int read;
    do
    {
        fm >> read;
        py_rowCSR.push_back(read);
        if(fm.fail())
            break;
    }while(!fm.eof());
}

template <typename T>
void readCSR(std::string file,
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
