#include <iostream>
#include <fstream>
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
             std::vector<int> &col_indices, std::vector<T> &values);
void COO_to_CSR(std::vector<int> &row_CSR,
                std::vector<int> row_COO, unsigned long Size, int matrixRow);

int main()
{
    int matrixRow = 30;
    Size = readCOO<float>(file, row_COO, col_IND, values);
    S_csrColInd = (int*)malloc(Size * sizeof(int));
    S_csrVal = (float*)malloc(Size * sizeof(float));
    COO_to_CSR(row_CSR, row_COO, Size, matrixRow);
    S_csrRowPtr = (int*)malloc(row_CSR.size() * sizeof(int));
    std::copy(row_CSR.begin(), row_CSR.end(), S_csrRowPtr);
    std::copy(col_IND.begin(), col_IND.end(), S_csrColInd);
    std::copy(values.begin(), values.end(), S_csrVal);
    for(int i = 0;i < row_CSR.size();i++)
    {
        std::cout << S_csrRowPtr[i] << " ";
    }
    std::cout << std::endl;
    for(int i = 0;i < col_IND.size();i++)
    {
        std::cout << S_csrColInd[i] << " ";
    }
    std::cout << std::endl;
    for(int i = 0;i < values.size();i++)
    {
        std::cout << S_csrVal[i] << " ";
    }
    free(S_csrVal);
    free(S_csrRowPtr);
    free(S_csrColInd);
    return 0;
}

//test for
template <typename T>
unsigned long readCOO(std::string file, std::vector<int> &row_indices,
             std::vector<int> &col_indices, std::vector<T> &values)
{
    int col_element, row_element;
    float value;
    std::ifstream fm(file, std::ios::in);
    if(!fm)
        std::cerr << "cannot open the file!\n";
    else
    {
        do
        {
            fm >> row_element;
            fm >> col_element;
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
    row_CSR.push_back(0);
    for(int i = 0;i < (Size - 1) && row_COO[i] <= row_COO[i + 1];i++)
    {
        if(row_COO[i] == row_COO[i + 1] + 1)
        {
            row_CSR.push_back(i + 1);
        }
        else if(row_COO[i] < row_COO[i + 1] + 1)
        {
            for(int j = 0;j < (row_COO[i + 1]- row_COO[i]);j++)
            {
                row_CSR.push_back(i + 1);
            }
        }
    }
    while(row_CSR.size() < matrixRow + 1)
    {
        row_CSR.push_back(row_CSR.back());
    }
}


