#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>


using namespace std;//think
using namespace chrono;


// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;


// const
void writeMatricesToFile(const Matrix* a, const Matrix*  b, const Matrix* c,const Matrix* c_cpu) {
    cout<<"hi"<<endl;
    ofstream outFile("output_p2.txt", ios::out);//,const char* filename
    if (!outFile) {
        cerr << "Error opening output file." << endl;
        return;
    }
    // a
    outFile << " INPUT Matrix (A):\n";
    for (int i = 0; i < a->height; ++i) {
        for (int j = 0; j < a->width; ++j) {
            outFile<< a->elements[i * a->width + j] <<" ";
            }
        outFile << "\n";
    }
    outFile << "\n";

    outFile << " kernel Matrix (b):\n";
    for (int i = 0; i < b->height; ++i) {
        for (int j = 0; j < b->width; ++j) {
            outFile<< b->elements[i * b->width + j] <<" ";
            }
        outFile << "\n";
    }
    outFile << "\n";

    outFile << " output Matrix (c_cuda):\n";
    for (int i = 0; i < c->height; ++i) {
        for (int j = 0; j < c->width; ++j) {
            outFile<< c->elements[i * c->width + j] <<" ";
        }
        outFile << "\n";
    }
    outFile << "\n";


    outFile << " output Matrix (c_cpu):\n";
    for (int i = 0; i < c_cpu->height; ++i) {
        for (int j = 0; j < c_cpu->width; ++j) {
            outFile<< c_cpu->elements[i * c_cpu->width + j] <<" ";
        }
        outFile << "\n";
    }
    outFile << "\n";

    outFile.close();
}


// Function to compute the L2,1 norm for two matrices
double computeL21Norm(const Matrix& matrix1, const Matrix& matrix2) {
    double l21Norm = 0.0;

    // Iterate over each column
    for (int j = 0; j < matrix1.width; ++j) {
        double colNorm = 0.0;

        // Iterate over each row in the column
        for (int i = 0; i < matrix1.height; ++i) {
            // Calculate the residual and add its square to colNorm
            double residual = matrix1.elements[i * matrix1.width + j] - matrix2.elements[i * matrix2.width + j];
            colNorm += residual * residual;
        }

        // Add the square root of colNorm to l21Norm
        l21Norm += sqrt(colNorm);
    }

    return l21Norm;
}

// Function to perform convolution without padding using CPU
void convolutionAndPadding_cpu(Matrix& inputMatrix, const Matrix& kernel, Matrix& outputMatrix, bool padding) {

    int kernelSize = kernel.height;

    if (padding){
    int paddingSizeTop = kernelSize / 2;
    int paddingSizeBottom = kernelSize - paddingSizeTop - 1;
    int paddedMatrixSize = inputMatrix.height + paddingSizeTop + paddingSizeBottom;
    float* paddedInputMatrix =(float*)malloc(paddedMatrixSize * paddedMatrixSize * sizeof(float));

    if (paddedInputMatrix == NULL) {
        printf("Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    // Initialize memory with 0.0
    memset(paddedInputMatrix, 0.0, paddedMatrixSize * paddedMatrixSize * sizeof(float));

    for (int i = paddingSizeTop; i < paddedMatrixSize - paddingSizeBottom; ++i) {
        for (int j = paddingSizeTop; j < paddedMatrixSize - paddingSizeBottom; ++j) {
            paddedInputMatrix[i * paddedMatrixSize + j] = inputMatrix.elements[(i - paddingSizeTop) * inputMatrix.width + (j - paddingSizeTop)];
        }
    }
    // now change inputMatrix
    inputMatrix.width =paddedMatrixSize;
    inputMatrix.height=paddedMatrixSize;
    inputMatrix.elements=paddedInputMatrix;


    }

    outputMatrix.width=inputMatrix.width-kernel.width+1;
    outputMatrix.height=inputMatrix.height-kernel.height+1;
    // vector<vector<float>> output(, vector<float>());
    outputMatrix.elements = (float*)malloc(outputMatrix.height * outputMatrix.width * sizeof(float));
    int inputRows = inputMatrix.height;
    int inputCols = inputMatrix.width;
    int outputSize = outputMatrix.height;//inputRows - kernelSize + 1;
    //square matrix assumption
    float* output = outputMatrix.elements;
    
    // Perform convolution
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            float sum = 0.0f;
            for (int ki = 0; ki < kernelSize; ++ki) {
                for (int kj = 0; kj < kernelSize; ++kj) {
                    // Calculate the indices for accessing elements in row-major order
                    // inputIdx,kernelIdx

                    sum += inputMatrix.elements[(i + ki) * inputCols + (j + kj)] * kernel.elements[ki * kernelSize + kj];
                }
            }
            output[i * outputSize + j] = sum; // Store the result in the output matrix
        }
    }
    
    return;
}

void maxPooling_cpu(Matrix& inputMatrix, int poolSize, Matrix& outputMatrix){
    int inputSize = inputMatrix.height;
    int outputSize = (inputSize + poolSize - 1) / poolSize;
    outputMatrix.width = outputSize;
    outputMatrix.height = outputSize;
    outputMatrix.elements = (float*)malloc(outputSize * outputSize * sizeof(float));
    for (int i = 0; i < outputSize; ++i){
        for (int j = 0; j < outputSize; ++j){
            int startRow = i * poolSize;
            int startCol = j * poolSize;
            int endRow = min(startRow + poolSize, inputSize);
            int endCol = min(startCol + poolSize, inputSize);
            float maxValue = inputMatrix.elements[startRow * inputSize + startCol];
            for (int row = startRow; row < endRow; ++row){
                for (int col = startCol; col < endCol; ++col){
                    maxValue = max(maxValue, inputMatrix.elements[row * inputSize + col]);
                }
            }
            outputMatrix.elements[i * outputSize + j] = maxValue;
        }
    }
}

void avgPooling_cpu(Matrix& inputMatrix, int poolSize, Matrix& outputMatrix){
    int inputSize = inputMatrix.height;
    int outputSize = (inputSize + poolSize - 1) / poolSize;
    outputMatrix.width = outputSize;
    outputMatrix.height = outputSize;
    outputMatrix.elements = (float*)malloc(outputSize * outputSize * sizeof(float));
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            int startRow = i * poolSize;
            int startCol = j * poolSize;
            int endRow = min(startRow + poolSize, inputSize);
            int endCol = min(startCol + poolSize, inputSize);
            float sum = 0.0;
            int count = 0;
            for (int row = startRow; row < endRow; ++row){
                for (int col = startCol; col < endCol; ++col){
                    sum += (inputMatrix.elements[row * inputSize + col]);
                    count++;
                }
            }
            outputMatrix.elements[i * outputSize + j] = sum/count;
        }
    }
}


float* convertToRowMajor(vector<vector<float>>& matrix, int rows, int cols) {
    float* rowMajor = (float*)malloc(rows * cols * sizeof(float));

    if (rowMajor == NULL) {
        printf("Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            rowMajor[i * cols + j] = matrix[i][j];
        }
    }

    return rowMajor;
}

// ReLU activation function
float relu(float x){
    return max(0.0f, x);
}

// tanh activation function
float tanh_activation(float x){
    return tanh(x);
}

// Apply ReLU activation function
vector<vector<float>> applyReLU(const vector<vector<float>>& inputMatrix){
    vector<vector<float>> outputMatrix(inputMatrix.size(), vector<float>(inputMatrix[0].size()));
    for (int i = 0; i < inputMatrix.size(); ++i) {
        for (int j = 0; j < inputMatrix[0].size(); ++j) {
            outputMatrix[i][j] = relu(inputMatrix[i][j]);
        }                                                                                                                                                   
    }
    return outputMatrix;
}

vector<vector<float>> applytanh(const vector<vector<float>>& inputMatrix){
    vector<vector<float>> outputMatrix(inputMatrix.size(), vector<float>(inputMatrix[0].size()));
    for (int i = 0; i < inputMatrix.size(); ++i) {
        for (int j = 0; j < inputMatrix[0].size(); ++j) {
            outputMatrix[i][j] = tanh_activation(inputMatrix[i][j]);
        }                                                                                                                                                   
    }
    return outputMatrix;
}

// Softmax function
vector<float> softmax(const vector<float>& inputVector){
    vector<float> probabilities(inputVector.size());
    float sumExp = 0.0;
    // Compute the sum of exponentials of input values
    for (float value : inputVector) {
        sumExp += exp(value);
    }
    // Compute softmax probabilities
    for (int i = 0; i < inputVector.size(); ++i) {
        probabilities[i] = exp(inputVector[i]) / sumExp;
    }
    return probabilities;
}

// Sigmoid function
vector<float> sigmoid(const vector<float>& inputVector){
    vector<float> probabilities(inputVector.size());
    // Compute sigmoid probabilities
    for (int i = 0; i < inputVector.size(); ++i){
        probabilities[i] = 1 / (1 + exp(-inputVector[i]));
    }

    return probabilities;
}


// Thread block size
#define BLOCK_SIZE 4

// Forward declaration of the matrix multiplication kernel
__global__ void convolutionAndPadding_cuda(const Matrix, const Matrix, Matrix);
__global__ void maxpooling_cuda(const Matrix,int poolsize,Matrix);
__global__ void avgpooling_cuda(const Matrix,Matrix);


// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix& A, const Matrix& B, Matrix& C)
{
    // cout<< "A dimn "<<A.width<<" "<<A.height<<endl;
    //output_cuda
    C.width=A.width-B.width+1;
    C.height=A.height-B.height+1;
    // vector<vector<float>> output(, vector<float>());
    C.elements = (float*)malloc(C.height * C.width * sizeof(float));


    cudaError_t cudaStatus;
    int deviceCount=1;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        // Print device properties
        printf("Device %d:\n", i);
        printf("  Name: %s\n", deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total global memory: %zu bytes\n", deviceProp.totalGlobalMem);
        // Add more properties as needed
    }


    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    auto t1 = high_resolution_clock::now(); //timer start
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((C.width+dimBlock.x-1)/ dimBlock.x, (C.height+dimBlock.y-1) / dimBlock.y);
    convolutionAndPadding_cuda<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "MatMulKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle error (e.g., clean up resources and return)
    }

    // Wait for kernel to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle error (e.g., clean up resources and return)
    }
    auto t2 = high_resolution_clock::now(); //timer start
    cout << "Time taken by cuda kernel computation: " << duration_cast<microseconds>(t2 - t1).count() << " microseconds\n";

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}


void hmpool(const Matrix& A, int poolsize , Matrix& C)
{
    // cout<< "A dimn "<<A.width<<" "<<A.height<<endl;
    //outputppoolmatrix_cuda

    //given input matrix and poolsize what will be the output pool size
    C.width = (A.width + poolsize -1)/poolsize;
    C.height = (A.height + poolsize -1)/poolsize;
    C.elements = (float*)malloc(C.height * C.width * sizeof(float));

    cudaError_t cudaStatus;
    int deviceCount=1;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        // Print device properties
        printf("Device %d:\n", i);
        printf("  Name: %s\n", deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total global memory: %zu bytes\n", deviceProp.totalGlobalMem);
        // Add more properties as needed
    }

    // Load A to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    auto t1 = high_resolution_clock::now(); //timer start
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((C.width+dimBlock.x-1)/ dimBlock.x, (C.height+dimBlock.y-1) / dimBlock.y);
    maxpooling_cuda<<<dimGrid, dimBlock>>>(d_A,poolsize,d_C);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "MatMulKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle error (e.g., clean up resources and return)
    }
    // Wait for kernel to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle error (e.g., clean up resources and return)
    }
    auto t2 = high_resolution_clock::now(); //timer start
    cout << "Time taken by cuda kernel computation: " << duration_cast<microseconds>(t2 - t1).count() << " microseconds\n";

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_C.elements);
}

void hapool(const Matrix& A, int poolsize , Matrix& C)
{
    // cout<< "A dimn "<<A.width<<" "<<A.height<<endl;
    //outputppoolmatrix_cuda

    //given input matrix and poolsize what will be the output pool size
    C.width = (A.width + poolsize -1)/poolsize;
    C.height = (A.height + poolsize -1)/poolsize;
    C.elements = (float*)malloc(C.height * C.width * sizeof(float));

    cudaError_t cudaStatus;
    int deviceCount=1;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        // Print device properties
        printf("Device %d:\n", i);
        printf("  Name: %s\n", deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total global memory: %zu bytes\n", deviceProp.totalGlobalMem);
        // Add more properties as needed
    }

    // Load A to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    auto t1 = high_resolution_clock::now(); //timer start
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((C.width+dimBlock.x-1)/ dimBlock.x, (C.height+dimBlock.y-1) / dimBlock.y);
    maxpooling_cuda<<<dimGrid, dimBlock>>>(d_A,poolsize,d_C);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "MatMulKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle error (e.g., clean up resources and return)
    }
    // Wait for kernel to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle error (e.g., clean up resources and return)
    }
    auto t2 = high_resolution_clock::now(); //timer start
    cout << "Time taken by cuda kernel computation: " << duration_cast<microseconds>(t2 - t1).count() << " microseconds\n";

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_C.elements);
}



// Matrix multiplication kernel called by MatMul()
__global__ void convolutionAndPadding_cuda(Matrix A, Matrix B, Matrix C)//(const Matrix& A, const Matrix& B, Matrix& C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row<C.height && col< C.width ){
        for (int i = 0; i < B.height; i++){
            for(int j=0; j< B.width; j++){
                // if (row + i < A.height && col + j < A.width){
                Cvalue += A.elements[(row+i)* A.width + (col +j)] 
                            *B.elements[i * B.width + j];

            }
        }
        
        C.elements[row * C.width + col] = Cvalue;
    }
}

// c.size = (A.size + poolSize - 1) / poolSize;
__global__ void maxpooling_cuda(const Matrix A, int poolsize,Matrix C) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < C.height && col < C.width){
        int startRow = row * poolsize;
        int startCol = col * poolsize;
        int endRow = min(startRow + poolsize, A.height);
        int endCol = min(startCol + poolsize, A.width);
        float maxValue = A.elements[startRow * A.width + startCol];
        for (int i = startRow; i < endRow; ++i) {
            for (int j = startCol; j < endCol; ++j) {
                maxValue = max(maxValue, A.elements[i * A.width + j]);
            }
        }
        C.elements[row * C.width + col] = maxValue;
    }
}

__global__ void avgpooling_cuda(const Matrix A, int poolsize, Matrix C) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < C.height && col < C.width){
        int startRow = row * poolsize;
        int startCol = col * poolsize;
        int endRow = min(startRow + poolsize, A.height);
        int endCol = min(startCol + poolsize, A.width);
        float sum = 0.0;
        int count = 0;
        for (int i = startRow; i < endRow; ++i) {
            for (int j = startCol; j < endCol; ++j) {
                sum += ( A.elements[i * A.width + j]);
                count++;
            }
        }
        C.elements[row * C.width + col] = sum/count;
    }
}

int main(int argc, char *argv[]) {

     if (argc < 2) {
        cout << "Usage: " << argv[0] << " <subtask_number> [subtask_arguments...]" << endl;
        return 1;
    }

    int subtask = stoi(argv[1]);

    switch (subtask) {
        case 1: {
            if (argc < 3){
                cout << "Usage: " << argv[0] << " 1 <N> <M> <P> <matrix_elements...>" << endl;
                return 1;
            }
            int N = stoi(argv[2]);
            int M = stoi(argv[3]);
            int P = stoi(argv[4]);
            vector<vector<float>> inputMatrix(N, vector<float>(N));
            for (int i = 0; i < N; ++i){
                for (int j = 0; j < N; ++j){
                    inputMatrix[i][j] = stof(argv[5 + i*N + j]);
                }
            }
            // Create a square kernel matrix
            vector<vector<float>> kernel(M, vector<float>(M));
            // Initialize kernel matrix
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < M; ++j) {
                    kernel[i][j] = stof(argv[5+N*N + i*M +j]);
                }
            }
            Matrix A;
            Matrix B;
            A.width =N;
            A.height=N;
            A.elements = convertToRowMajor(inputMatrix, A.height,A.width);

            B.width =M;
            B.height=M;
            B.elements = convertToRowMajor(kernel, B.height,B.width);

            Matrix C_cpu;
            Matrix C_cuda;
            
            bool padding;
            if (P == 0){
                padding = false;
                auto t1 = high_resolution_clock::now(); //timer start
                convolutionAndPadding_cpu(A, B, C_cpu, padding);
                auto t2 = high_resolution_clock::now(); //timerstop
                MatMul(A,B,C_cuda);
                auto t3 = high_resolution_clock::now(); //timerstop
                cout << "Time taken by cuda whole computation: " << duration_cast<microseconds>(t3 - t2).count() << " microseconds\n";
                cout << "Time taken by cpu computation: " << duration_cast<microseconds>(t2-t1).count() << " microseconds\n";
                double l21norm =computeL21Norm(C_cuda,C_cpu);
                cout<< "L2,1 norm of residual matrix (C_cuda-C_cpu)= "<<l21norm<<endl;
            }
            else{
                padding = true;
                auto t1 = high_resolution_clock::now(); //timer start
                convolutionAndPadding_cpu(A, B, C_cpu, padding);
                auto t2 = high_resolution_clock::now(); //timerstop
                MatMul(A,B,C_cuda);
                auto t3 = high_resolution_clock::now(); //timerstop
                cout << "Time taken by cuda whole computation: " << duration_cast<microseconds>(t3 - t2).count() << " microseconds\n";
                cout << "Time taken by cpu computation: " << duration_cast<microseconds>(t2-t1).count() << " microseconds\n";
                double l21norm =computeL21Norm(C_cuda,C_cpu);
                cout<< "L2,1 norm of residual matrix (C_cuda-C_cpu)= "<<l21norm<<endl;
            }
            // cout << "Output Matrix:" << endl;
            // printMatrix(outputMatrix);
            cout<<endl;
            break;
        }
        case 2: {
            if (argc < 3){
                cout << "Usage: " << argv[0] << " 2 <activation_function> <N> <M> <matrix_elements...>" << endl;
                return 1;
            }
            int activationFunction = stoi(argv[2]);
            int N = stoi(argv[3]);
            int M = stoi(argv[4]);
            vector<vector<float>> inputMatrix(N, vector<float>(M));
            for (int i = 0; i < N; ++i){
                for (int j = 0; j < M; ++j){
                    inputMatrix[i][j] = stof(argv[5 + i*M + j]);
                }
            }
            
            vector<vector<float>> outputMatrix;
            if (activationFunction == 0)
                outputMatrix = applyReLU(inputMatrix);
            else if (activationFunction == 1)
                outputMatrix = applytanh(inputMatrix); 

            // cout << "Output Matrix:" << endl;
            //printMatrix(outputMatrix);
            cout<<endl;
            break;
        }
        case 3: {
            if (argc < 3){
                cout << "Usage: " << argv[0] << " 3 <pooling_function> <N> <matrix_elements...>" << endl;
                return 1;
            }
            int poolingFunction = stoi(argv[2]);
            int N = stoi(argv[3]);
            vector<vector<float>> inputMatrix(N, vector<float>(N));
            for (int i = 0; i < N; ++i){
                for (int j = 0; j < N; ++j){
                    inputMatrix[i][j] = stof(argv[4 + i*N + j]);
                }
            }

            Matrix A;
            A.width =N;
            A.height=N;
            A.elements = convertToRowMajor(inputMatrix, A.height,A.width);

            Matrix C_cpu;
            Matrix C_cuda;

            if (poolingFunction == 0){
                auto t1 = high_resolution_clock::now(); //timer start
                maxPooling_cpu(A, 2, C_cpu);
                // A shape will change according to padding
                auto t2 = high_resolution_clock::now(); //timerstop
                hmpool(A, 2,C_cuda);
                auto t3 = high_resolution_clock::now(); //timerstop
                cout << "Time taken by cuda whole computation: " << duration_cast<microseconds>(t3 - t2).count() << " microseconds\n";
                cout << "Time taken by cpu computation: " << duration_cast<microseconds>(t2-t1).count() << " microseconds\n";

                double l21norm =computeL21Norm(C_cuda,C_cpu);
                cout<< "L2,1 norm of residual matrix (C_cuda-C_cpu)= "<<l21norm<<endl;
            }
            else if (poolingFunction == 1){
                auto t1 = high_resolution_clock::now(); //timer start
                avgPooling_cpu(A, 2, C_cpu);
                // A shape will change according to padding
                auto t2 = high_resolution_clock::now(); //timerstop
                hapool(A, 2, C_cuda);
                auto t3 = high_resolution_clock::now(); //timerstop
                cout << "Time taken by cuda whole computation: " << duration_cast<microseconds>(t3 - t2).count() << " microseconds\n";
                cout << "Time taken by cpu computation: " << duration_cast<microseconds>(t2-t1).count() << " microseconds\n";

                double l21norm =computeL21Norm(C_cuda,C_cpu);
                cout<< "L2,1 norm of residual matrix (C_cuda-C_cpu)= "<<l21norm<<endl;
            }
            // cout << "Output Matrix:" << endl;
            // /printMatrix(outputMatrix);
            //cout<<endl;
            break;
        }
        case 4: {
            if (argc < 3) {
                cout << "Usage: " << argv[0] << " 4 <function_type> <vector_elements...>" << endl;
                return 1;
            }
            int functionType = stoi(argv[2]);
            vector<float> inputVector;
            for (int i = 3; i < argc; ++i) {
                inputVector.push_back(stof(argv[i]));
            }
            vector<float> outputVector;
            if (functionType == 0)
                outputVector = sigmoid(inputVector);
            else if (functionType == 1)
                outputVector = softmax(inputVector);

            // cout << "Output Vector:" << endl;
            for (float val : outputVector) {
                cout << val << " ";
            }
            cout << endl;
            break;
        }
        default:
            cout << "Invalid subtask number." << endl;
            return 1;
    }

    //writeMatricesToFile(&A,&B,&C_cuda,&C_cpu);//,"output_device.txt");
    

    return 1;
}