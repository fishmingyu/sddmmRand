#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>

std::string file = "/Users/fishming/Documents/数据与算法/pythonLearn/sparse.txt";
std::vector<int> row_COO, col_IND, row_CSR;
std::vector<float> values;

int* S_csrRowPtr = nullptr, * S_csrColInd = nullptr;
float* S_csrVal = nullptr;
unsigned long Size;

template <typename T>
unsigned long readCOO(std::string file, std::vector<int> &row_indices,
             std::vector<int> &col_indices, std::vector<T> &values,
                      int &matrixRow, int &matrixCol);
void COO_to_CSR(std::vector<int> &row_CSR,
                std::vector<int> row_COO, unsigned long Size, int matrixRow);

int main()
{
    int matrixRow, matrixCol;
    Size = readCOO<float>(file, row_COO, col_IND, values, matrixRow, matrixCol);
    S_csrColInd = (int*)malloc(Size * sizeof(int));
    S_csrVal = (float*)malloc(Size * sizeof(float));
    std::sort(row_COO.begin(), row_COO.end());
    COO_to_CSR(row_CSR, row_COO, Size, matrixRow);
    S_csrRowPtr = (int*)malloc(row_CSR.size() * sizeof(int));
    std::copy(row_CSR.begin(), row_CSR.end(), S_csrRowPtr);
    std::copy(col_IND.begin(), col_IND.end(), S_csrColInd);
    std::copy(values.begin(), values.end(), S_csrVal);
    for(int i = 0;i < row_COO.size();i++)
    {
        std::cout << row_COO[i] << " ";
    }
    std::cout << std::endl;
    for(int i = 0;i < row_CSR.size();i++)
    {
        std::cout << S_csrRowPtr[i] << " ";
    }
    if(row_CSR.size() != matrixRow + 1)
        std::cout << "error!" << std::endl;
    std::cout << std::endl;
    free(S_csrVal);
    free(S_csrRowPtr);
    free(S_csrColInd);
    return 0;
}

//test for
template <typename T>
unsigned long readCOO(std::string file, std::vector<int> &row_indices,
             std::vector<int> &col_indices, std::vector<T> &values,
                      int &matrixRow, int &matrixCol)
{
    int col_element, row_element;
    float value;
    std::ifstream fm(file, std::ios::in);
    if(!fm)
        std::cerr << "cannot open the file!\n";
    else
    {
        fm >> matrixRow >> matrixCol;
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
    return col_indices.size();
}


void COO_to_CSR(std::vector<int> &row_CSR, std::vector<int> row_COO,
                unsigned long Size, int matrixRow)
{
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
    for(int j = 0;j < matrixRow + 1- row_COO.back();j++)
    {
        row_CSR.push_back(static_cast<int>(Size));
    }
}


