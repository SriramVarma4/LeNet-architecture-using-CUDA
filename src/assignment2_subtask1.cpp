#include <iostream>
#include <vector>
#include <random>
using namespace std;

//convolution(no padding)
vector<vector<float>> convolutionWithoutPadding(const vector<vector<float>>& inputMatrix, const vector<vector<float>>& kernel,int size) {
    int inputRows = inputMatrix.size();
    int inputCols = inputMatrix[0].size();
    int kernelSize = kernel.size();
    int outputSize = inputRows - kernelSize + 1;
    vector<vector<float>> output(outputSize, vector<float>(outputSize, 0.0));   
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            for (int ki = 0; ki < kernelSize; ++ki) {
                for (int kj = 0; kj < kernelSize; ++kj) {
                    output[i][j] += inputMatrix[i + ki][j + kj] * kernel[ki][kj];
                }
            }
        }
    }
    return output;
}

//convolution with padding
vector<vector<float>> convolutionWithPadding(const vector<vector<float>>& inputMatrix, const vector<vector<float>>& kernel){
    int kernelSize = kernel.size();
    int pt = kernelSize / 2;
    int pb = kernelSize - pt - 1;
    int outputSize = inputMatrix.size() + pt + pb;

    vector<vector<float>> paddedInputMatrix(outputSize, vector<float>(outputSize, 0.0));
    for (int i = pt; i< outputSize-pb; ++i){
        for (int j = pt;j< outputSize-pb; ++j){
            paddedInputMatrix[i][j] = inputMatrix[i - pt][j - pt];
        }
    }

    vector<vector<float>> outputMatrix(inputMatrix.size(), vector<float>(inputMatrix[0].size(), 0.0));
    for (int i = 0; i < outputMatrix.size(); ++i){
        for (int j = 0; j < outputMatrix[0].size(); ++j){
            for (int ki = 0; ki < kernelSize; ++ki){
                for (int kj = 0; kj < kernelSize; ++kj){
                    outputMatrix[i][j] += paddedInputMatrix[i + ki][j + kj] * kernel[ki][kj];
                }
            }
        }
    }
    return outputMatrix;
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

// maxPooling
vector<vector<float>> maxPooling(const vector<vector<float>>& inputMatrix, int poolSize){
    int inputSize = inputMatrix.size();
    int outputSize = (inputSize + poolSize - 1) / poolSize;
    vector<vector<float>> outputMatrix(outputSize, vector<float>(outputSize));
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            float maxValue = inputMatrix[i * poolSize][j * poolSize];
            for (int k = 0; k < poolSize && i * poolSize + k < inputSize; ++k){
                for (int l = 0; l < poolSize && j * poolSize + l < inputSize; ++l){
                    maxValue = max(maxValue, inputMatrix[i * poolSize + k][j * poolSize + l]);
                }
            }
            outputMatrix[i][j] = maxValue;
        }
    }
    return outputMatrix;
}

// avgPooling
vector<vector<float>> averagePooling(const vector<vector<float>>& inputMatrix, int poolSize){
    int inputSize = inputMatrix.size();
    int outputSize = (inputSize + poolSize - 1) / poolSize;
    vector<vector<float>> outputMatrix(outputSize, vector<float>(outputSize));
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            float sum = 0.0;
            int count = 0;
            for (int k = 0; k < poolSize && i * poolSize + k < inputSize; ++k){
                for (int l = 0; l < poolSize && j * poolSize + l < inputSize; ++l){
                    sum += inputMatrix[i * poolSize + k][j * poolSize + l];
                    ++count;
                }
            }
            outputMatrix[i][j] = sum / count;
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

void printMatrix(const vector<vector<float>>& matrix){
    for (const auto& row : matrix){
        for (float val : row) {
            cout << val << " ";
        }
        //cout << endl;
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
            vector<vector<float>> outputMatrix;
            if (P == 0)
                outputMatrix = convolutionWithoutPadding(inputMatrix, kernel, N);
            else
                outputMatrix = convolutionWithPadding(inputMatrix, kernel);

            // cout << "Output Matrix:" << endl;
            printMatrix(outputMatrix);
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
            printMatrix(outputMatrix);
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
            vector<vector<float>> outputMatrix;
            if (poolingFunction == 0)
                outputMatrix = maxPooling(inputMatrix, 2);
            else if (poolingFunction == 1)
                outputMatrix = averagePooling(inputMatrix, 2);
            // cout << "Output Matrix:" << endl;
            printMatrix(outputMatrix);
            cout<<endl;
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

    return 0;
}