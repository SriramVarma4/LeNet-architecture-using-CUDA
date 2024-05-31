#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;//think
using namespace chrono;


// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    int depth;
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

float* vectorToRowMajor(vector<float>& matrix) {
    int n=matrix.size();
    float* rowMajor = (float*)malloc(n* sizeof(float));

    if (rowMajor == NULL) {
        printf("Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i=0;i<n;i++){
        rowMajor[i]=matrix[i];
    }

    return rowMajor;
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


// Thread block size
#define BLOCK_SIZE 4

// Forward declaration of kernel
__global__ void convolutionAndPadding_cuda_1d(const Matrix, const Matrix, Matrix);
__global__ void convolutionAndPadding_cuda_2d(const Matrix, const Matrix, Matrix);
__global__ void maxpooling_cuda_2d(const Matrix, Matrix);
__global__ void convolution_cuda_3d(const Matrix, const Matrix, Matrix);
__global__ void fc1_cuda_3d(const Matrix, const Matrix, Matrix);
__global__ void fc2_cuda_3d(const Matrix, const Matrix, Matrix);



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

    // // 1d kernel>
    // auto t1 = high_resolution_clock::now(); //timer start
    // // Invoke kernel
    // dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 dimGrid((C.width+dimBlock.x-1)/ dimBlock.x, (C.height+dimBlock.y-1) / dimBlock.y);
    // convolutionAndPadding_cuda_1d<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    // cudaStatus = cudaGetLastError();
    // if (cudaStatus != cudaSuccess) {
    //     fprintf(stderr, "MatMulKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    //     // Handle error (e.g., clean up resources and return)
    // }

    // // Wait for kernel to finish
    // cudaStatus = cudaDeviceSynchronize();
    // if (cudaStatus != cudaSuccess) {
    //     fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
    //     // Handle error (e.g., clean up resources and return)
    // }
    // auto t2 = high_resolution_clock::now(); //timer start
    // cout << "Time taken by cuda kernel computation: " << duration_cast<microseconds>(t2 - t1).count() << " microseconds\n";

    // // 1d kernel <

    // 2d kernel >
    auto t1 = high_resolution_clock::now(); //timer start
    // Invoke kernel
    dim3 dimBlock(4,4,4);
    dim3 dimGrid((C.width+dimBlock.x-1)/ dimBlock.x, (C.height+dimBlock.y-1) / dimBlock.y,(C.depth+dimBlock.z-1) / dimBlock.z);
    convolutionAndPadding_cuda_2d<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "convolutionAndPadding_cuda_2d launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
    // 2d kernel <

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void convolutionAndPadding_cuda_1d(Matrix A, Matrix B, Matrix C)//(const Matrix& A, const Matrix& B, Matrix& C)
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


//  kernel called by MatMul()
__global__ void convolutionAndPadding_cuda_2d(Matrix A, Matrix B, Matrix C)//(const Matrix& A, const Matrix& B, Matrix& C)
{
    // Each thread computes one element of C at (x,y,z) -- (i,j,ck)
    // computation is for zth kernel's (x,y) location 

    int id_y = blockIdx.y * blockDim.y + threadIdx.y;
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_z=blockIdx.z * blockDim.z+ threadIdx.z;

    // zth kernel base indices
    int bk =(B.width*B.height)*id_z;
    int ck =(C.width*C.height)*id_z;

    //bias term for zth kernel
    float bias =B.elements[B.width*B.height*C.depth+id_z];
    
    // by accumulating results into Cvalue
    float Cvalue = 0;
    if (id_y<C.height && (id_x < C.width && id_z < C.depth ) ){
        for (int i = 0; i < B.width; i++){
            for(int j=0; j< B.height; j++){
                // if (row + i < A.height && col + j < A.width){

                Cvalue += A.elements[(id_y+j)* A.width + (id_x +i)] 
                            *B.elements[bk+(j * B.width) + i];

            }
        }
        
        C.elements[ck +(id_y * C.width)+ id_x] = Cvalue+bias;
    }
}

// c.size = (A.size + poolSize - 1) / poolSize;
// numbe of kernels =C.depth
__global__ void maxpooling_cuda_2d(const Matrix A, Matrix C) 
{
// Each thread computes one element of C at (x,y,z) -- (i,j,ck)
    // computation is for zth kernel's (x,y) location 

    int id_y = blockIdx.y * blockDim.y + threadIdx.y;
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_z=blockIdx.z * blockDim.z+ threadIdx.z;

    // zth kernel base indices
    int ck =(C.width*C.height)*id_z;

    if (id_y<C.height && (id_x < C.width && id_z < C.depth ) ){
        int poolsize=2;
        int startRow = id_y * poolsize;
        int startCol = id_x * poolsize;
        int endRow = min(startRow + poolsize, A.height);
        int endCol = min(startCol + poolsize, A.width);
        float maxValue = A.elements[startRow * A.width + startCol];
        for (int j = startRow; j < endRow; ++j) {
            for (int i = startCol; i < endCol; ++j){
                maxValue =max(maxValue,A.elements[j* A.width + i]);
            }
        }
        
        C.elements[ck +(id_y * C.width)+ id_x] =maxValue;
    }
}


//  kernel called by MatMul()
// here B is 3d conv operator/kernel  
// numbe of kernels =C.depth
__global__ void convolution_cuda_3d(Matrix A, Matrix B, Matrix C)//(const Matrix& A, const Matrix& B, Matrix& C)
{
    // Each thread computes one element of C at (x,y,z) -- (i,j,ck)
    // computation is for zth kernel's (x,y) location 

    int id_y = blockIdx.y * blockDim.y + threadIdx.y;
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_z=blockIdx.z * blockDim.z+ threadIdx.z;

    // zth kernel base indices
    int bz =(B.width*B.height*B.depth)*id_z;
    int cz =(C.width*C.height)*id_z;
    // int az =

    //bias term for zth kernel
    float bias =B.elements[B.width*B.height*B.depth*C.depth+id_z];
    
    if (id_y<C.height && (id_x < C.width && id_z < C.depth ) ){
        // by accumulating results into Cvalue
        float Cvalue = 0;
        for (int k = 0; k < B.depth; k++){
            int bk =bz+(B.height*B.width*k);
            int ak =(A.height*A.width*k);
            for (int i = 0; i < B.width; i++){
                for(int j=0; j< B.height; j++){
                    // if (row + i < A.height && col + j < A.width){

                    Cvalue += A.elements[ak+((id_y+j)* A.width) + (id_x +i)] 
                                *B.elements[bk+(j * B.width) + i];

                }
            }
        
    }
    C.elements[cz +(id_y * C.width)+ id_x] = Cvalue+bias;
}
}

//  kernel called by MatMul()
// here B is 3d conv operator/kernel  
// numbe of kernels =C.depth
__global__ void fc2_cuda_3d(Matrix A, Matrix B, Matrix C)//(const Matrix& A, const Matrix& B, Matrix& C)
{
    // Each thread computes one element of C at (x,y,z) -- (i,j,ck)
    // computation is for zth kernel's (x,y) location 

    int id_y = blockIdx.y * blockDim.y + threadIdx.y;
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_z=blockIdx.z * blockDim.z+ threadIdx.z;

    // zth kernel base indices
    int bz =(B.width*B.height*B.depth)*id_z;
    int cz =(C.width*C.height)*id_z;
    // int az =

    //bias term for zth kernel
    float bias =B.elements[B.width*B.height*B.depth*C.depth+id_z];
    
    if (id_y<C.height && (id_x < C.width && id_z < C.depth ) ){
        // by accumulating results into Cvalue
        float Cvalue = 0;
        for (int k = 0; k < B.depth; k++){
            int bk =bz+(B.height*B.width*k);
            int ak =(A.height*A.width*k);
            for (int i = 0; i < B.width; i++){
                for(int j=0; j< B.height; j++){
                    // if (row + i < A.height && col + j < A.width){

                    Cvalue += A.elements[ak+((id_y+j)* A.width) + (id_x +i)] 
                                *B.elements[bk+(j * B.width) + i];

                }
            }
        
    }
    C.elements[cz +(id_y * C.width)+ id_x] = Cvalue+bias;
}
}

//  kernel called by MatMul()
// here B is 3d conv operator/kernel  
// numbe of kernels =C.depth
__global__ void fc1_cuda_3d(Matrix A, Matrix B, Matrix C)//(const Matrix& A, const Matrix& B, Matrix& C)
{
    // Each thread computes one element of C at (x,y,z) -- (i,j,ck)
    // computation is for zth kernel's (x,y) location 

    int id_y = blockIdx.y * blockDim.y + threadIdx.y;
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_z=blockIdx.z * blockDim.z+ threadIdx.z;

    // zth kernel base indices
    int bz =(B.width*B.height*B.depth)*id_z;
    int cz =(C.width*C.height)*id_z;
    // int az =

    //bias term for zth kernel
    float bias =B.elements[B.width*B.height*B.depth*C.depth+id_z];
    
    if (id_y<C.height && (id_x < C.width && id_z < C.depth ) ){
        // by accumulating results into Cvalue
        float Cvalue = 0;
        for (int k = 0; k < B.depth; k++){
            int bk =bz+(B.height*B.width*k);
            int ak =(A.height*A.width*k);
            for (int i = 0; i < B.width; i++){
                for(int j=0; j< B.height; j++){
                    // if (row + i < A.height && col + j < A.width){

                    Cvalue += A.elements[ak+((id_y+j)* A.width) + (id_x +i)] 
                                *B.elements[bk+(j * B.width) + i];

                }
            }
        
    }
    C.elements[cz +(id_y * C.width)+ id_x] = max((Cvalue+bias),0.0f);//relu(x)
}
}

float* readWeightsBiases(const string& filename) {
    ifstream file(filename);
    vector<float> values;

    float* rowMajor=nullptr;
    if (!file.is_open()) {
        cerr << "Error opening file " << filename << endl;
    return rowMajor;
    }

    float value;
    while (file >> value) {
        values.push_back(value);
    }

    file.close();

    //vectortoRowMajor
    int n=values.size();
    rowMajor = (float*)malloc(n* sizeof(float));

    if (rowMajor == NULL) {
        printf("Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i=0;i<n;i++){
        rowMajor[i]=values[i];
    }

    return rowMajor;
}

 
void readImageToVector(string filePath , Matrix& image) {//const&
    float* pixelValues=image.elements;
    ifstream file(filePath);
    if (!file.is_open()) {
        cerr << "Failed to open file." << endl;
        return ;
    }
    string line;
    istringstream iss(line);
    // float temp;
     for (int j = 0; j < 28; ++j) {
    for (int i = 0; i < 28; ++i) {

            getline(file, line);
            pixelValues[j*image.width+i]=stof(line);
            //cout<<pixelValues[i][j]<<" ";
        }
       // cout<<" "<<endl;
    }

    return ;
}


int findMaxIndex(float* vec) {
    float maxVal = vec[0];
    int maxIndex = 0;

    for (int i = 1; i < 10; i++) {
        if (vec[i] > maxVal) {
            maxVal = vec[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}

string extractName(const string& imagePath) {
    size_t pos = imagePath.rfind('/');
    if (pos != string::npos) {
        return imagePath.substr(pos + 1);
    }
    return ""; // Return empty string if '/' is not found
}

void  printtop5(float* probabilities, string image_path){
    vector<vector<float>> nums;

    string output_path="/scratch/cse/dual/cs5190439/col380_work/output/"+extractName(image_path);
    ofstream outputFile(output_path);

    for(int i=0; i<10; i++){
        nums.push_back({probabilities[i],float(i) });
    }
    sort(nums.begin(), nums.end());
    string temp;
    for(int i=9; i>4; i--){
        temp=to_string(nums[i][0])+" class"+to_string( int(nums[i][1])) +"\n";
        outputFile<<temp;

       // cout<<nums[i][0]<<" "<<"class"<<nums[i][1]<<endl;
    }
    //cout<<image_path<<endl;

}

void softmax(float* inputVector, float* probabilities, string image_path) {
    //vector<float> probabilities(inputVector.size());
    float sumExp = 0.0;

    // float max_ele= *max_element(inputVector.begin(), inputVector.end());
    float max_ele=inputVector[0];
    for (int i=0;i<10;i++){
        max_ele=max(max_ele,inputVector[i]);
    }

    // cout<<"Softmax -- "<<endl;
    // for (float value : inputVector) {
    //     cout<<value<<" ";
        
    // }
    // cout<<""<<endl;

    for (int i=0;i<10;i++){
        probabilities[i]=inputVector[i];
       
    }

    // for (float value : probabilities) {
    //     cout<<value<<" ";
    // }
    // cout<<""<<endl;

    // Compute the sum of exponentials of input values
    for (int i=0;i<10;i++){
        sumExp += exp(probabilities[i]);
    }
    //cout<<"sumEXP "<< sumExp<<endl;

    // Compute softmax probabilities
    for (int i = 0; i < 10; i++) {
        probabilities[i] = exp(probabilities[i]) / sumExp;
        probabilities[i]=probabilities[i]*100.0;
    }
    //------------for printing top5
     printtop5(probabilities,image_path);

    return ;
}


int actual(string image_path){
    return image_path[image_path.length()-5]-'0';
}

int predict_folder(string folder_path, string folder_text){

    float correct=0.0;
    float total=0.0;
    ifstream file(folder_text);
    
    //conv1
    // vector<float> values = readWeightsBiases("trained_weights/conv1.txt");
    // float* conv1 =vectorToRowMajor(conv1_values);
    float* conv_1 = readWeightsBiases("/scratch/cse/dual/cs5190439/col380_work/data/trained_weights/conv1.txt");

    Matrix conv1;
    conv1.height =5;
    conv1.width=5;
    conv1.depth =20;
    conv1.elements=conv_1;

    // conv2
    // values = readWeightsBiases("trained_weights/conv2.txt");
    float* conv_2 = readWeightsBiases("/scratch/cse/dual/cs5190439/col380_work/data/trained_weights/conv2.txt");

    Matrix conv2;
    conv2.height =5;
    conv2.width=5;
    conv2.depth =20;
    conv2.elements=conv_2;
    
    //fc1
    // values = readWeightsBiases("trained_weights/fc1.txt");
    float* fc_1 = readWeightsBiases("/scratch/cse/dual/cs5190439/col380_work/data/trained_weights/fc1.txt");

    Matrix fc1;
    fc1.height =4;
    fc1.width=4;
    fc1.depth =50;
    fc1.elements=fc_1;

    // fc2
    float* fc_2 = readWeightsBiases("/scratch/cse/dual/cs5190439/col380_work/data/trained_weights/fc2.txt");

    Matrix fc2;
    fc2.height =1;
    fc2.width=1;
    fc2.depth =500;
    fc2.elements=fc_2;

    //image read
    

    auto t1 = high_resolution_clock::now(); //timer start

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


     // Invoke kernel
    dim3 dimBlock(4,4,4);
    int poolSize=2;


    //image_1

    Matrix image;
    image.height=28;
    image.width=28;
    image.elements= (float*)malloc(28*28*sizeof(float));

    Matrix d_image;
    d_image.width = 28; d_image.height =28;
    size_t size = 28*28 * sizeof(float);
    cudaMalloc(&d_image.elements, size);

    //load pretrained weights on cuda
    Matrix d_conv1;
    d_conv1.width = conv1.width; d_conv1.height = conv1.height; d_conv1.depth =conv1.depth;

    size = 520* sizeof(float);
    cudaMalloc(&d_conv1.elements, size);
    cudaMemcpy(d_conv1.elements,conv1.elements, size,
               cudaMemcpyHostToDevice);

    Matrix d_conv2;
    d_conv2.width = conv2.width; d_conv2.height = conv2.height;d_conv2.depth =conv2.depth;

    size = 25050* sizeof(float);
    cudaMalloc(&d_conv2.elements, size);
    cudaMemcpy(d_conv2.elements, conv2.elements, size,
               cudaMemcpyHostToDevice);

    Matrix d_fc1;
    d_fc1.width = fc1.width; d_fc1.height = fc1.height; d_fc1.depth =fc1.depth;

    size = 400500* sizeof(float);
    cudaMalloc(&d_fc1.elements, size);
    cudaMemcpy(d_fc1.elements, fc1.elements, size,
               cudaMemcpyHostToDevice);

    Matrix d_fc2;
    d_fc2.width = fc2.width; d_fc2.height = fc2.height;d_fc2.depth =fc2.depth;

    size = 5010* sizeof(float);
    cudaMalloc(&d_fc2.elements, size);
    cudaMemcpy(d_fc2.elements, fc2.elements, size,
               cudaMemcpyHostToDevice);


    // conv1
    // Allocate C_conv1 -conv1_out in device memory
    Matrix d_C_conv1;
    d_C_conv1.width = 24; d_C_conv1.height = 24;d_C_conv1.depth =20;
    size = d_C_conv1.width * d_C_conv1.height * d_C_conv1.depth* sizeof(float);
    cudaMalloc(&d_C_conv1.elements, size);

    dim3 dimGrid_conv1((d_C_conv1.width+dimBlock.x-1)/ dimBlock.x, (d_C_conv1.height+dimBlock.y-1) / dimBlock.y,(d_C_conv1.depth+dimBlock.z-1) / dimBlock.z);

    
    // pool1
    // Allocate C_pool1 -pool1_out in device memory
    Matrix d_C_pool1;
    d_C_pool1.width = 12; d_C_pool1.height = 12;d_C_pool1.depth =20;
    size = d_C_pool1.width * d_C_pool1.height * d_C_pool1.depth* sizeof(float);
    cudaMalloc(&d_C_pool1.elements, size);
    dim3 dimGrid_pool1((d_C_pool1.width+dimBlock.x-1)/ dimBlock.x, (d_C_pool1.height+dimBlock.y-1) / dimBlock.y,(d_C_pool1.depth+dimBlock.z-1) / dimBlock.z);

    // conv2
    // Allocate C_conv2 -conv2_out in device memory
    Matrix d_C_conv2;
    d_C_conv2.width =8; d_C_conv2.height = 8;d_C_conv2.depth =50;
    size = d_C_conv2.width * d_C_conv2.height * d_C_conv2.depth* sizeof(float);
    cudaMalloc(&d_C_conv2.elements, size);

    dim3 dimGrid_conv2((d_C_conv2.width+dimBlock.x-1)/ dimBlock.x, (d_C_conv2.height+dimBlock.y-1) / dimBlock.y,(d_C_conv2.depth+dimBlock.z-1) / dimBlock.z);
  
    // pool2
    // Allocate C_pool2 -pool2_out in device memory
    Matrix d_C_pool2;
    // int poolSize=2;
    d_C_pool2.width = 4; d_C_pool2.height = 4;d_C_pool2.depth =50;
    size = d_C_pool2.width * d_C_pool2.height * d_C_pool2.depth* sizeof(float);
    cudaMalloc(&d_C_pool2.elements, size);
    dim3 dimGrid_pool2((d_C_pool2.width+dimBlock.x-1)/ dimBlock.x, (d_C_pool2.height+dimBlock.y-1) / dimBlock.y,(d_C_pool2.depth+dimBlock.z-1) / dimBlock.z);

    // fc1
    // Allocate C_fc1 -fc1_out in device memory
    Matrix d_C_fc1;
    d_C_fc1.width = 1; d_C_fc1.height = 1;d_C_fc1.depth =500;
    size = d_C_fc1.width * d_C_fc1.height * d_C_fc1.depth* sizeof(float);
    cudaMalloc(&d_C_fc1.elements, size);
    dim3 dimGrid_fc1((d_C_fc1.width+dimBlock.x-1)/ dimBlock.x, (d_C_fc1.height+dimBlock.y-1) / dimBlock.y,(d_C_fc1.depth+dimBlock.z-1) / dimBlock.z);

    // fc2
    // Allocate C_fc2 -fc2_out in device memory
    Matrix d_C_fc2;
    d_C_fc2.width = 1; d_C_fc2.height = 1;d_C_fc2.depth =10;
    size = d_C_fc2.width * d_C_fc2.height * d_C_fc2.depth* sizeof(float);
    cudaMalloc(&d_C_fc2.elements, size);
    dim3 dimGrid_fc2((d_C_fc2.width+dimBlock.x-1)/ dimBlock.x, (d_C_fc2.height+dimBlock.y-1) / dimBlock.y,(d_C_fc2.depth+dimBlock.z-1) / dimBlock.z);

    //image_2
    Matrix image2;
    image2.height=28;
    image2.width=28;
    image2.elements= (float*)malloc(28*28*sizeof(float));
    
    Matrix d_image2;
    d_image2.width = 28; d_image2.height =28;
    size_t size = 28*28 * sizeof(float);
    cudaMalloc(&d_image2.elements, size);

    //load pretrained weights on cuda
    Matrix d_conv12;
    d_conv12.width = conv1.width; d_conv12.height = conv1.height; d_conv12.depth =conv1.depth;

     size = 520* sizeof(float);
    cudaMalloc(&d_conv12.elements, size);
    cudaMemcpy(d_conv12.elements,conv1.elements, size,
               cudaMemcpyHostToDevice);

    Matrix d_conv22;
    d_conv22.width = conv2.width; d_conv22.height = conv2.height;d_conv22.depth =conv2.depth;

    size = 25050* sizeof(float);
    cudaMalloc(&d_conv22.elements, size);
    cudaMemcpy(d_conv22.elements, conv2.elements, size,
               cudaMemcpyHostToDevice);

    Matrix d_fc12;
    d_fc12.width = fc1.width; d_fc12.height = fc1.height; d_fc12.depth =fc1.depth;

    size = 400500* sizeof(float);
    cudaMalloc(&d_fc12.elements, size);
    cudaMemcpy(d_fc12.elements, fc1.elements, size,
               cudaMemcpyHostToDevice);

    Matrix d_fc22;
    d_fc22.width = fc2.width; d_fc22.height = fc2.height;d_fc22.depth =fc2.depth;

    size = 5010* sizeof(float);
    cudaMalloc(&d_fc22.elements, size);
    cudaMemcpy(d_fc22.elements, fc2.elements, size,
               cudaMemcpyHostToDevice);


    // conv1
    // Allocate C_conv1 -conv1_out in device memory
    Matrix d_C_conv12;
    d_C_conv12.width = 24; d_C_conv12.height = 24;d_C_conv12.depth =20;
    size = d_C_conv12.width * d_C_conv12.height * d_C_conv12.depth* sizeof(float);
    cudaMalloc(&d_C_conv12.elements, size);

    dim3 dimGrid_conv12((d_C_conv12.width+dimBlock.x-1)/ dimBlock.x, (d_C_conv12.height+dimBlock.y-1) / dimBlock.y,(d_C_conv12.depth+dimBlock.z-1) / dimBlock.z);

    // pool1
    // Allocate C_pool1 -pool1_out in device memory
    Matrix d_C_pool12;
    d_C_pool12.width = 12; d_C_pool12.height = 12;d_C_pool12.depth =20;
    size = d_C_pool12.width * d_C_pool12.height * d_C_pool12.depth* sizeof(float);
    cudaMalloc(&d_C_pool12.elements, size);
    dim3 dimGrid_pool12((d_C_pool12.width+dimBlock.x-1)/ dimBlock.x, (d_C_pool12.height+dimBlock.y-1) / dimBlock.y,(d_C_pool12.depth+dimBlock.z-1) / dimBlock.z);

    // conv2
    // Allocate C_conv2 -conv2_out in device memory
    Matrix d_C_conv22;
    d_C_conv22.width =8; d_C_conv22.height = 8;d_C_conv22.depth =50;
    size = d_C_conv22.width * d_C_conv22.height * d_C_conv22.depth* sizeof(float);
    cudaMalloc(&d_C_conv22.elements, size);

    dim3 dimGrid_conv22((d_C_conv22.width+dimBlock.x-1)/ dimBlock.x, (d_C_conv22.height+dimBlock.y-1) / dimBlock.y,(d_C_conv22.depth+dimBlock.z-1) / dimBlock.z);
  
    // pool2
    // Allocate C_pool2 -pool2_out in device memory
    Matrix d_C_pool22;
    // int poolSize=2;
    d_C_pool22.width = 4; d_C_pool22.height = 4;d_C_pool22.depth =50;
    size = d_C_pool22.width * d_C_pool22.height * d_C_pool22.depth* sizeof(float);
    cudaMalloc(&d_C_pool22.elements, size);
    dim3 dimGrid_pool22((d_C_pool22.width+dimBlock.x-1)/ dimBlock.x, (d_C_pool22.height+dimBlock.y-1) / dimBlock.y,(d_C_pool22.depth+dimBlock.z-1) / dimBlock.z);

    // fc1
    // Allocate C_fc1 -fc1_out in device memory
    Matrix d_C_fc12;
    d_C_fc12.width = 1; d_C_fc12.height = 1;d_C_fc12.depth =500;
    size = d_C_fc12.width * d_C_fc12.height * d_C_fc12.depth* sizeof(float);
    cudaMalloc(&d_C_fc12.elements, size);
    dim3 dimGrid_fc12((d_C_fc12.width+dimBlock.x-1)/ dimBlock.x, (d_C_fc12.height+dimBlock.y-1) / dimBlock.y,(d_C_fc12.depth+dimBlock.z-1) / dimBlock.z);

    // fc2
    // Allocate C_fc2 -fc2_out in device memory
    Matrix d_C_fc22;
    d_C_fc22.width = 1; d_C_fc22.height = 1;d_C_fc22.depth =10;
    size = d_C_fc22.width * d_C_fc22.height * d_C_fc22.depth* sizeof(float);
    cudaMalloc(&d_C_fc22.elements, size);
    dim3 dimGrid_fc22((d_C_fc22.width+dimBlock.x-1)/ dimBlock.x, (d_C_fc22.height+dimBlock.y-1) / dimBlock.y,(d_C_fc22.depth+dimBlock.z-1) / dimBlock.z);

    float* output10 = (float*)malloc(10*sizeof(float));

    float* probabilities = (float*)malloc(10*sizeof(float));
    

    float* output102 = (float*)malloc(10*sizeof(float));

    float* probabilities2 = (float*)malloc(10*sizeof(float));

    if (!file.is_open()) {
        cerr << "Failed to open file -folder.txt." << endl;
        return 1;
    }

    string filename;
    string image_path;
    float a,b;
    int c=0;

    int streams_no = 2;
    //float a0,a1,a2,a3,a4,a5;

    while (getline(file, filename)) 
    {
       // cout << "Filename: " << folder_path+"/"+filename << endl;
        image_path=folder_path+"/"+filename;
        cout<<"file: "<<image_path<<endl;

        vector<<string<<paths;

        paths.push_back(image_path)
        if(paths.size()==streams_no){
            cudaStream_t stream[streams_no];

            for(int i=0; i<streams_no; i++) {
                cudaStreamCreate(&stream[i]);
            }

            vector<float> actual;
            for(int i=0;i<paths.size();i++){
                a= actual(paths[i]);
                actual.push_back(a);
            }

            //read image -28x28
            readImageToVector(paths[0], image);
            cudaMemcpy(d_image.elements, image.elements, size,
                    cudaMemcpyHostToDevice);
            readImageToVector(paths[1], image2);
            cudaMemcpy(d_image2.elements, image2.elements, size,
                    cudaMemcpyHostToDevice);

            convolutionAndPadding_cuda_2d<<<dimGrid_conv1, dimBlock,0,stream0>>>(d_image, d_conv1, d_C_conv1);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "convolutionAndPadding_cuda_2d launch failed: %s\n", cudaGetErrorString(cudaStatus));
                // Handle error (e.g., clean up resources and return)
            }
            convolutionAndPadding_cuda_2d<<<dimGrid_conv12, dimBlock,0,stream1>>>(d_image2, d_conv12, d_C_conv12);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "convolutionAndPadding_cuda_2d launch failed: %s\n", cudaGetErrorString(cudaStatus));
                // Handle error (e.g., clean up resources and return)
            }   

            cudaStreamSynchronize(stream0);
            cudaStreamSynchronize(stream1);


            maxpooling_cuda_2d<<<dimGrid_pool1, dimBlock,0,stream0>>>(d_C_conv1, d_C_pool1);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "maxpooling_cuda_2d launch failed: %s\n", cudaGetErrorString(cudaStatus));
                // Handle error (e.g., clean up resources and return)
            } 

            maxpooling_cuda_2d<<<dimGrid_pool12, dimBlock,0,stream1>>>(d_C_conv12, d_C_pool12);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "maxpooling_cuda_2d launch failed: %s\n", cudaGetErrorString(cudaStatus));
                // Handle error (e.g., clean up resources and return)
            } 

            cudaStreamSynchronize(stream0);
            cudaStreamSynchronize(stream1);  

            convolution_cuda_3d<<<dimGrid_conv2, dimBlock,0,stream0>>>(d_C_pool1, d_conv2, d_C_conv2);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "convolution_cuda_3d launch failed: %s\n", cudaGetErrorString(cudaStatus));
                // Handle error (e.g., clean up resources and return)
            }

            convolution_cuda_3d<<<dimGrid_conv22, dimBlock,0,stream1>>>(d_C_pool12, d_conv22, d_C_conv22);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "convolution_cuda_3d launch failed: %s\n", cudaGetErrorString(cudaStatus));
                // Handle error (e.g., clean up resources and return)
            }

            cudaStreamSynchronize(stream0);
            cudaStreamSynchronize(stream1);


            maxpooling_cuda_2d<<<dimGrid_pool2, dimBlock,0,stream0>>>(d_C_conv2, d_C_pool2);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "maxpooling_cuda_2d launch failed: %s\n", cudaGetErrorString(cudaStatus));
                // Handle error (e.g., clean up resources and return)
            }

            maxpooling_cuda_2d<<<dimGrid_pool22, dimBlock,0,stream1>>>(d_C_conv22, d_C_pool22);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "maxpooling_cuda_2d launch failed: %s\n", cudaGetErrorString(cudaStatus));
                // Handle error (e.g., clean up resources and return)
            }

            cudaStreamSynchronize(stream0);
            cudaStreamSynchronize(stream1);

            fc1_cuda_3d<<<dimGrid_fc1, dimBlock,0,stream0>>>(d_C_pool2, d_fc1, d_C_fc1);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "fc1_cuda_3d launch failed: %s\n", cudaGetErrorString(cudaStatus));
                // Handle error (e.g., clean up resources and return)
            }

            fc1_cuda_3d<<<dimGrid_fc12, dimBlock,0,stream1>>>(d_C_pool22, d_fc12, d_C_fc12);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "fc1_cuda_3d launch failed: %s\n", cudaGetErrorString(cudaStatus));
                // Handle error (e.g., clean up resources and return)
            }

            cudaStreamSynchronize(stream0);
            cudaStreamSynchronize(stream1);

            fc2_cuda_3d<<<dimGrid_fc2, dimBlock,0,stream0>>>(d_C_pool2, d_fc2, d_C_fc2);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "fc2_cuda_3d launch failed: %s\n", cudaGetErrorString(cudaStatus));
                // Handle error (e.g., clean up resources and return)
            }

            fc2_cuda_3d<<<dimGrid_fc22, dimBlock,0,stream1>>>(d_C_pool22, d_fc22, d_C_fc22);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "fc2_cuda_3d launch failed: %s\n", cudaGetErrorString(cudaStatus));
                // Handle error (e.g., clean up resources and return)
            }

            cudaStreamSynchronize(stream0);
            cudaStreamSynchronize(stream1);

            // Read final out from device memory
            cudaMemcpy(output10, d_C_fc2.elements, size,
                    cudaMemcpyDeviceToHost);
            
            cudaMemcpy(output102, d_C_fc22.elements, size,
                    cudaMemcpyDeviceToHost);

            // for(int i=0; i<fc2_output.size(); i++){
            //     cout<<fc2_output[i]<<" ";
            // }
            // cout<<""<<endl;

            
            softmax(output10, probabilities,paths[0]); 
            softmax(output102, probabilities2,paths[1]); 

        
            // for(int i=0; i<fc2_output.size(); i++){
            //     cout<<softmax_output[i]<<" ";
            // }
            // cout<<" "<<endl;

            a1 = findMaxIndex(probabilities);
            a1 = findMaxIndex(probabilities2);
            

            // remove it to get control to cpu 
            if(actual[0]==a1){
                correct++;
            }
            total++;

            if(actual[1]==a2){
                correct++;
            }
            total++;
            c=total;
            if(c%100==0){
                cout<<"total: "<<total <<"Accuracy: "<<(correct/total)*100.0  <<endl;
                
            }

            paths.clear();

        }
        //a=actual(image_path);

        // CNN arch

        
        // readImageToVector(paths[0], image);

        
        // // conv1
        // convolutionAndPadding_cuda_2d<<<dimGrid_conv1, dimBlock>>>(d_image, d_conv1, d_C_conv1);
        // cudaStatus = cudaGetLastError();
        // if (cudaStatus != cudaSuccess) {
        //     fprintf(stderr, "convolutionAndPadding_cuda_2d launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //     // Handle error (e.g., clean up resources and return)
        // }

        // // Wait for kernel to finish
        // cudaStatus = cudaDeviceSynchronize();
        // if (cudaStatus != cudaSuccess) {
        //     fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        //     // Handle error (e.g., clean up resources and return)
        // }

        // pool1
        // maxpooling_cuda_2d<<<dimGrid_pool1, dimBlock>>>(d_C_conv1, d_C_pool1);
        // cudaStatus = cudaGetLastError();
        // if (cudaStatus != cudaSuccess) {
        //     fprintf(stderr, "maxpooling_cuda_2d launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //     // Handle error (e.g., clean up resources and return)
        // }


        // conv2
        // convolution_cuda_3d<<<dimGrid_conv2, dimBlock>>>(d_C_pool1, d_conv2, d_C_conv2);
        // cudaStatus = cudaGetLastError();
        // if (cudaStatus != cudaSuccess) {
        //     fprintf(stderr, "convolution_cuda_3d launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //     // Handle error (e.g., clean up resources and return)
        // }

        // pool2
        // maxpooling_cuda_2d<<<dimGrid_pool2, dimBlock>>>(d_C_conv2, d_C_pool2);
        // cudaStatus = cudaGetLastError();
        // if (cudaStatus != cudaSuccess) {
        //     fprintf(stderr, "maxpooling_cuda_2d launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //     // Handle error (e.g., clean up resources and return)
        // }

        // fc1
        // fc1_cuda_3d<<<dimGrid_fc1, dimBlock>>>(d_C_pool2, d_fc1, d_C_fc1);
        // cudaStatus = cudaGetLastError();
        // if (cudaStatus != cudaSuccess) {
        //     fprintf(stderr, "fc1_cuda_3d launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //     // Handle error (e.g., clean up resources and return)
        // }

        // fc2
        // fc2_cuda_3d<<<dimGrid_fc2, dimBlock>>>(d_C_pool2, d_fc2, d_C_fc2);
        // cudaStatus = cudaGetLastError();
        // if (cudaStatus != cudaSuccess) {
        //     fprintf(stderr, "fc2_cuda_3d launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //     // Handle error (e.g., clean up resources and return)
        // }

        // Read final out from device memory
        // cudaMemcpy(output10, d_C_fc2.elements, size,
        //         cudaMemcpyDeviceToHost);

        // // for(int i=0; i<fc2_output.size(); i++){
        // //     cout<<fc2_output[i]<<" ";
        // // }
        // // cout<<""<<endl;


        // softmax(output10, probabilities,image_path); 
    
        // // for(int i=0; i<fc2_output.size(); i++){
        // //     cout<<softmax_output[i]<<" ";
        // // }
        // // cout<<" "<<endl;

        // b = findMaxIndex(probabilities);

        // // remove it to get control to cpu 
        // if(a==b){
        //         correct++;
        // }
        // total++;
        // c=total;
        // if(c%100==0){
        //     cout<<"total: "<<total <<"Accuracy: "<<(correct/total)*100.0  <<endl;
            
        // }

        
       
    }   
    // Free device memory
    cudaFree(d_C_conv1.elements);
    cudaFree(d_C_conv2.elements);
    cudaFree(d_C_fc1.elements);
    cudaFree(d_C_fc1.elements);
    cudaFree(d_C_pool1.elements);
    cudaFree(d_C_pool2.elements);

    cudaFree(d_conv1.elements);
    cudaFree(d_conv2.elements);
    cudaFree(d_fc1.elements);
    cudaFree(d_fc2.elements);
    cudaFree(d_image.elements);

    cudaFree(d_C_conv12.elements);
    cudaFree(d_C_conv22.elements);
    cudaFree(d_C_fc12.elements);
    cudaFree(d_C_fc12.elements);
    cudaFree(d_C_pool12.elements);
    cudaFree(d_C_pool22.elements);

    cudaFree(d_conv12.elements);
    cudaFree(d_conv22.elements);
    cudaFree(d_fc12.elements);
    cudaFree(d_fc22.elements);
    cudaFree(d_image2.elements);

    auto t2 = high_resolution_clock::now(); //timer start
    cout << "Time taken by cuda kernel computation: " << duration_cast<microseconds>(t2 - t1).count() << " microseconds\n";
    // 2d kernel <
 
    file.close();
    float acc= (correct/total)*100.0 ;
    cout<<"total: "<<total <<endl;


    return acc;
    

}

int main() {
    //reading weights  conv1

    //cout<<"hi"<<endl;
    // string folder_path="processed_test";
    // string folder_text="precessed_test.txt";

    // cout<<accuracy(folder_path,folder_text)<<endl;

    //---------------------------------------------

    string folder_path="/scratch/cse/dual/cs5190439/col380_work/processed_test_neg";
    string folder_text="/scratch/cse/dual/cs5190439/col380_work/precessed_test_neg.txt";
    cout<<predict_folder(folder_path,folder_text)<<endl;



    //---------------------------------
    // string image_path="processed_test_case/2.txt";

    // cout<<predict(image_path) <<" " << actual(image_path)<<endl;
    
    return 0;
}

// int main() {

//     // Create a 16*16 input matrix with random values
//     int n =16;
//     int k =4;
//     vector<vector<float>> inputMatrix(n, vector<float>(n));
//     random_device rd;
//     mt19937 gen(rd());
//     uniform_real_distribution<float> dis(0.0, 1.0);
//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < n; ++j) {
//             inputMatrix[i][j] = dis(gen);
//         }
//     }

//     // Create a 3x3 kernel with random values
//     vector<vector<float>> kernel(k, vector<float>(k));
//     for (int i = 0; i < k; ++i) {
//         for (int j = 0; j < k; ++j) {
//             kernel[i][j] = dis(gen);
//         }
//     }
//     Matrix A;
//     Matrix B;
//     A.width =n;
//     A.height=n;
//     A.elements = convertToRowMajor(inputMatrix, A.height,A.width);

//     B.width =k;
//     B.height=k;
//     B.elements = convertToRowMajor(kernel, B.height,B.width);

//     bool padding =true;

//     //output_cpu - C_cpu
//     Matrix C_cpu;
//     Matrix C_cuda;

//     auto t1 = high_resolution_clock::now(); //timer start
//     convolutionAndPadding_cpu(A, B, C_cpu, padding);
//     // A shape will change according to padding
//     auto t2 = high_resolution_clock::now(); //timerstop
//     MatMul(A,B,C_cuda);
//     auto t3 = high_resolution_clock::now(); //timerstop
//     cout << "Time taken by cuda whole computation: " << duration_cast<microseconds>(t3 - t2).count() << " microseconds\n";
//     cout << "Time taken by cpu computation: " << duration_cast<microseconds>(t2-t1).count() << " microseconds\n";

//     double l21norm =computeL21Norm(C_cuda,C_cpu);
//     cout<< "L2,1 norm of residual matrix (C_cuda-C_cpu)= "<<l21norm<<endl;

//     writeMatricesToFile(&A,&B,&C_cuda,&C_cpu);//,"output_device.txt");
    

//     return 1;
// }
