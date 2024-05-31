#include <iostream>
#include <fstream>
#include <vector>
#include <bits/stdc++.h>
#include <filesystem>
#include <random>

using namespace std;
 
void readImageToVector(const string& filePath , vector<vector<float>> &pixelValues) {
    ifstream file(filePath);
    if (!file.is_open()) {
        cerr << "Failed to open file." << endl;
        return ;
    }
    string line;
    istringstream iss(line);
    float temp;
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            getline(file, line);
            pixelValues[i][j]=stof(line);
            //cout<<pixelValues[i][j]<<" ";
        }
       // cout<<" "<<endl;
    }

    return ;
}

vector<float> readWeightsBiases(const string& filename) {
    ifstream file(filename);
    vector<float> values;

    if (!file.is_open()) {
        cerr << "Error opening file " << filename << endl;
        return values;
    }

    float value;
    while (file >> value) {
        values.push_back(value);
    }

    file.close();
    return values;
}

vector<vector<float>> convolutionWithoutPaddingBias(const vector<vector<float>>& inputMatrix, const vector<vector<float>>& kernel, 
float Bias) {
    
    vector<vector<float>> output(24, vector<float>(24, 0.0));    
    for (int i = 0; i < 24; i++) {
        for (int j = 0; j < 24; j++) {

            for (int ki = 0; ki < 5; ki++) {
                for (int kj = 0; kj < 5; kj++) {
                    output[i][j] += inputMatrix[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            
            output[i][j]=output[i][j]+Bias;
            
        }
    }
    
    return output;
}


vector<vector<float>> convolutionWithoutPaddingBias_conv2(const vector<vector<vector<float>>>& inputMatrix, const vector<vector<vector<float>>>& kernel, 
float Bias) {

    vector<vector<float>> output(8, vector<float>(8, 0.0));
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j <8; j++) {

                for (int ki = 0; ki < 5; ki++) {
                    for (int kj = 0; kj < 5; kj++) {
                            for(int kk=0; kk<20; kk++){
                                output[i][j] += inputMatrix[kk][i + ki][j + kj] * kernel[kk][ki][kj];
                            }
                    }

                }
            
            //adding bias
            output[i][j]=output[i][j]+Bias;
            
        }
    }
    
    return output;
}


vector<vector<float>> convolutionWithoutPaddingBias_fc1(const vector<vector<vector<float>>>& inputMatrix, const vector<vector<vector<float>>>& kernel, 
float Bias) {

    vector<vector<float>> output(1, vector<float>(1));       
    for (int ki = 0; ki < 4; ki++) {
         for (int kj = 0; kj < 4 ; kj++) {
            for(int kk=0; kk<50; kk++){
                output[0][0] += inputMatrix[kk][ ki][kj] * kernel[kk][ki][kj];
                }
            }
    }
            
    //adding bias
    output[0][0]+=Bias;

    return output;
}

float convolutionWithoutPaddingBias_fc2(const vector<vector<vector<float>>>& inputMatrix, const vector<vector<vector<float>>>& kernel, 
float Bias) {

    float output=0.0;      
    for (int ki = 0; ki < 1; ki++) {
         for (int kj = 0; kj < 1 ; kj++) {
            for(int kk=0; kk<500; kk++){
                output += inputMatrix[kk][ ki][kj] * kernel[kk][ki][kj];
                }
            }
    }
    output+=Bias;
    return output;
}


// Function to perform max pooling on a square input matrix
vector<vector<float>> maxPooling(const vector<vector<float>>& inputMatrix) {
    int poolSize=2;
    int inputSize = inputMatrix.size();
    int outputSize = (inputSize ) / poolSize;
    vector<vector<float>> outputMatrix(outputSize, vector<float>(outputSize));
    
    float maxValue = 0.0;
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < outputSize; j++) {

           maxValue=inputMatrix[i * poolSize][j * poolSize];
            
            for (int k = 0; k < poolSize ; k++) {
                for (int l = 0; l < poolSize ; l++) {
                    if(inputMatrix[i * poolSize + k][j * poolSize + l] > maxValue){
                        maxValue=inputMatrix[i * poolSize + k][j * poolSize + l] ;
                    }
                }
            }
            outputMatrix[i][j] = maxValue;
        }
    }

    return outputMatrix;
}


void conv1(const vector<vector<float>>& inputMatrix, vector<vector<vector<float>>> &output
,  vector<vector<vector<float>>> &kernels , vector<float> &biases){

    //input:: 28x28
    //output: 20x24x24
    
    for(int k=0; k<20; k++){
        output[k]= convolutionWithoutPaddingBias(inputMatrix,kernels[k],biases[k]);
    }

    return;
}

void pool( vector<vector<vector<float>>> & inputMatrix, vector<vector<vector<float>>>&output ){
    //input: 20x24x24 || 50x8x8
    //output: 20x12x12 || 50x4x4

    //vector<vector<vector<float>>> output(inputMatrix.size(),vector<vector<float>>(inputMatrix.size()/2, vector<float> (inputMatrix.size()/2)) );
    
    for(int k=0; k<inputMatrix.size(); k++){
        output[k]= maxPooling(inputMatrix[k]);
    }

    return ;
}

void conv2(const vector<vector<vector<float>>>& inputMatrix, vector<vector<vector<float>>> &output,
vector<vector<vector<vector<float>>>> &kernels,   vector<float> &biases){
    //input: 20x12x12
    //output: 50x8x8
    //kernal:  50x  20x5x5
    //bias: 50
  
    //vector<vector<vector<float>>> output(50,vector<vector<float>>(8, vector<float> (8)) );
    
    for(int k=0; k<50; k++){
        output[k]= convolutionWithoutPaddingBias_conv2(inputMatrix,kernels[k],biases[k]);
    }

    return ;
}

float relu(float x) {
    return max(0.0f, x);
}
void fc1(const vector<vector<vector<float>>>& inputMatrix,vector<vector<vector<float>>> &output,
   vector<vector<vector<vector<float>>>> &kernels,vector<float> &biases){
    //input: 50x4x4
    //output: 500x1x1
    //kernal:  500x  50x4x4
    //bias: 500


    
    for(int k=0; k<500; k++){
        output[k]= convolutionWithoutPaddingBias_fc1(inputMatrix,kernels[k],biases[k]);
    }

    for(int i=0; i<500; i++){
        for(int j=0; j<1; j++){
            for(int k=0; k<1; k++){
                output[i][j][k]=relu(output[i][j][k]);
            }
        }
    }


    return;
}



void fc2(const vector<vector<vector<float>>>& inputMatrix, vector<float> &output,
vector<vector<vector<vector<float>>>> &kernels, vector<float> &biases){
    //input: 500x1x1
    //output: 10x1
    //kernal:  10x  500x1x1
    //bias: 50
   
    //vector<float> output(10 );
    
    for(int k=0; k<10; k++){
        output[k]= convolutionWithoutPaddingBias_fc2(inputMatrix,kernels[k],biases[k]);
    }

    return ;
}

string extractName(const string& imagePath) {
    size_t pos = imagePath.rfind('/');
    if (pos != string::npos) {
        return imagePath.substr(pos + 1);
    }
    return ""; // Return empty string if '/' is not found
}
 void  printtop5(vector<float> probabilities, string image_path){
    vector<vector<float>> nums;

    string output_path="output/"+extractName(image_path);
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

// Softmax function
void softmax(vector<float>& inputVector, vector<float> & probabilities, string image_path) {
    //vector<float> probabilities(inputVector.size());
    float sumExp = 0.0;

    float max_ele= *max_element(inputVector.begin(), inputVector.end());
    // cout<<"Softmax -- "<<endl;
    // for (float value : inputVector) {
    //     cout<<value<<" ";
        
    // }
    // cout<<""<<endl;

    for(int i=0; i<inputVector.size(); i++){
        probabilities[i]=inputVector[i];
       
    }

    // for (float value : probabilities) {
    //     cout<<value<<" ";
    // }
    // cout<<""<<endl;

    // Compute the sum of exponentials of input values
    for (float value : probabilities) {
        sumExp += exp(value);
    }
    //cout<<"sumEXP "<< sumExp<<endl;

    // Compute softmax probabilities
    for (int i = 0; i < probabilities.size(); i++) {
        probabilities[i] = exp(probabilities[i]) / sumExp;
        probabilities[i]=probabilities[i]*100.0;
    }
    //------------for printing top5
     printtop5(probabilities,image_path);

    return ;
}



int findMaxIndex(const vector<float>& vec) {
    float maxVal = vec[0];
    int maxIndex = 0;

    for (int i = 1; i < vec.size(); ++i) {
        if (vec[i] > maxVal) {
            maxVal = vec[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}

int actual(string image_path){
    return image_path[image_path.length()-5]-'0';
}


int predict_folder(string image_path, vector<vector<float>> &image , vector<vector<vector<float>>> &conv1_output
,  vector<vector<vector<float>>> &pool1_output , vector<vector<vector<float>>> &conv2_output
,  vector<vector<vector<float>>> &pool2_output,  vector<vector<vector<float>>> &fc1_output,
   vector<float> &fc2_output,  vector<float> &softmax_output
   , vector<vector<vector<float>>> &conv1_kernels,vector<float> &conv1_biases
   ,  vector<vector<vector<vector<float>>>> &conv2_kernels,vector<float> &conv2_biases
   ,  vector<vector<vector<vector<float>>>> &fc1_kernels,vector<float> &fc1_biases
   ,  vector<vector<vector<vector<float>>>> &fc2_kernels,vector<float> &fc2_biases){
    
    
    //read image -28x28
    readImageToVector(image_path, image);
    //conv1 - 28x28 -> 20x24x24
    conv1(image,conv1_output,conv1_kernels,conv1_biases);

    //maxpool-20x24x24 -> 20x12x12
    pool(conv1_output, pool1_output);

    //conv2: 20x12x12-> 50x8x8
    conv2(pool1_output,conv2_output,conv2_kernels,conv2_biases);

    //pool2-50x8x8 -> 50x4x4
    pool(conv2_output,  pool2_output);

    // fc1:  50x4x4 -> 500x1x1
    fc1(pool2_output, fc1_output,fc1_kernels,fc1_biases);

    //fc2: 500x1x1 ->10
    fc2(fc1_output,  fc2_output,fc2_kernels,fc2_biases);


    // for(int i=0; i<fc2_output.size(); i++){
    //     cout<<fc2_output[i]<<" ";
    // }
    // cout<<""<<endl;

    
   softmax(fc2_output, softmax_output,image_path); 
    // for(int i=0; i<fc2_output.size(); i++){
    //     cout<<softmax_output[i]<<" ";
    // }
    // cout<<" "<<endl;

    return findMaxIndex(softmax_output);
}


int predict(string image_path){
    vector<float> values = readWeightsBiases("../weights/conv1.txt");
    int w=0;
    vector<vector<vector<float>>> conv1_kernels(20, vector<vector<float>>(5, vector<float>(5) ) );

    vector<float> conv1_biases(20 );
    for(int k=0 ; k<20; k++){
        for(int i=0; i<5; i++){
            for(int j=0; j<5; j++ ){
                conv1_kernels[k][i][j]=values[w];
                w++;
            }
        }
    }

    for(int k=0; k<20; k++){
        conv1_biases[k]=values[w];
        w++;
    }

    values = readWeightsBiases("../weights/conv2.txt");
    w=0;
    vector<vector<vector<vector<float>>>> conv2_kernels(50, vector<vector<vector<float>>>(20,
     vector<vector<float>>(5, vector<float>(5)) ) );

    vector<float> conv2_biases(50);

    for(int l=0; l<50; l++){
        for(int k=0 ; k<20; k++){
            for(int i=0; i<5; i++){
                for(int j=0; j<5; j++ ){
                    conv2_kernels[l][k][i][j]=values[w];
                    w++;
                }
            }
        }
    }

    for(int l=0; l<50; l++){
        conv2_biases[l]=values[w];
        w++;
    }

    values = readWeightsBiases("../weights/fc1.txt");
     w=0;
    vector<vector<vector<vector<float>>>> fc1_kernels(500, vector<vector<vector<float>>>(50,
     vector<vector<float>>(4, vector<float>(4)) ) );

    vector<float> fc1_biases(500 );

    for(int l=0; l<500; l++){
        for(int k=0 ; k<50; k++){
            for(int i=0; i<4; i++){
                for(int j=0; j<4; j++ ){
                    fc1_kernels[l][k][i][j]=values[w];
                    w++;
                }
            }
        }
    }

    for(int l=0; l<500; l++){
        fc1_biases[l]=values[w];
        w++;
    }

    values = readWeightsBiases("../weights/fc2.txt");
     w=0;
    vector<vector<vector<vector<float>>>> fc2_kernels( 10, vector<vector<vector<float>>>(500,
     vector<vector<float>>(1, vector<float>(1)) ) );

    vector<float> fc2_biases(10 );

    for(int l=0; l<10; l++){
        for(int k=0 ; k<500; k++){
            for(int i=0; i<1; i++){
                for(int j=0; j<1; j++ ){
                    fc2_kernels[l][k][i][j]=values[w];
                    w++;
                }
            }
        }
    }

    for(int l=0; l<10; l++){
        fc2_biases[l]=values[w];
        w++;
    }

    vector<vector<float>> image(28, vector<float>(28)); 
    vector<vector<vector<float>>> conv1_output(20 , vector<vector<float>>(24, vector<float>(24)) );
    vector<vector<vector<float>>> pool1_output(20 , vector<vector<float>>(12, vector<float>(12)) );
     vector<vector<vector<float>>> conv2_output(50 , vector<vector<float>>(8, vector<float>(8)) );
    vector<vector<vector<float>>> pool2_output(50 , vector<vector<float>>(4, vector<float>(4)) );
    vector<vector<vector<float>>> fc1_output(500 , vector<vector<float>>(1, vector<float>(1)) );
    vector<float> fc2_output(10);  
    vector<float> softmax_output(10);

    return predict_folder(image_path,image,conv1_output,pool1_output,conv2_output,pool2_output,fc1_output,fc2_output,softmax_output
       ,conv1_kernels,conv1_biases,conv2_kernels,conv2_biases,fc1_kernels,fc1_biases,fc2_kernels,fc2_biases);
}

float accuracy(string folder_path, string folder_text){
    float correct=0.0;
    float total=0.0;
    ifstream file(folder_text);
    
    vector<float> values = readWeightsBiases("../weights/conv1.txt");
    int w=0;
    vector<vector<vector<float>>> conv1_kernels(20, vector<vector<float>>(5, vector<float>(5) ) );

    vector<float> conv1_biases(20 );
    for(int k=0 ; k<20; k++){
        for(int i=0; i<5; i++){
            for(int j=0; j<5; j++ ){
                conv1_kernels[k][i][j]=values[w];
                w++;
            }
        }
    }

    for(int k=0; k<20; k++){
        conv1_biases[k]=values[w];
        w++;
    }

    values = readWeightsBiases("../weights/conv2.txt");
    w=0;
    vector<vector<vector<vector<float>>>> conv2_kernels(50, vector<vector<vector<float>>>(20,
     vector<vector<float>>(5, vector<float>(5)) ) );

    vector<float> conv2_biases(50 );

    for(int l=0; l<50; l++){
        for(int k=0 ; k<20; k++){
            for(int i=0; i<5; i++){
                for(int j=0; j<5; j++ ){
                    conv2_kernels[l][k][i][j]=values[w];
                    w++;
                }
            }
        }
    }

    for(int l=0; l<50; l++){
        conv2_biases[l]=values[w];
        w++;
    }

    values = readWeightsBiases("../weights/fc1.txt");
     w=0;
    vector<vector<vector<vector<float>>>> fc1_kernels(500, vector<vector<vector<float>>>(50,
     vector<vector<float>>(4, vector<float>(4)) ) );

    vector<float> fc1_biases(500 );

    for(int l=0; l<500; l++){
        for(int k=0 ; k<50; k++){
            for(int i=0; i<4; i++){
                for(int j=0; j<4; j++ ){
                    fc1_kernels[l][k][i][j]=values[w];
                    w++;
                }
            }
        }
    }

    for(int l=0; l<500; l++){
        fc1_biases[l]=values[w];
        w++;
    }

    values = readWeightsBiases("../weights/fc2.txt");
     w=0;
    vector<vector<vector<vector<float>>>> fc2_kernels( 10, vector<vector<vector<float>>>(500,
     vector<vector<float>>(1, vector<float>(1)) ) );

    vector<float> fc2_biases(10 );

    for(int l=0; l<10; l++){
        for(int k=0 ; k<500; k++){
            for(int i=0; i<1; i++){
                for(int j=0; j<1; j++ ){
                    fc2_kernels[l][k][i][j]=values[w];
                    w++;
                }
            }
        }
    }

    for(int l=0; l<10; l++){
        fc2_biases[l]=values[w];
        w++;
    }

    vector<vector<float>> image(28, vector<float>(28)); 
    vector<vector<vector<float>>> conv1_output(20 , vector<vector<float>>(24, vector<float>(24)) );
    vector<vector<vector<float>>> pool1_output(20 , vector<vector<float>>(12, vector<float>(12)) );
     vector<vector<vector<float>>> conv2_output(50 , vector<vector<float>>(8, vector<float>(8)) );
    vector<vector<vector<float>>> pool2_output(50 , vector<vector<float>>(4, vector<float>(4)) );
    vector<vector<vector<float>>> fc1_output(500 , vector<vector<float>>(1, vector<float>(1)) );
    vector<float> fc2_output(10);  
    vector<float> softmax_output(10);

    if (!file.is_open()) {
        cerr << "Failed to open file." << endl;
        return 1;
    }

    string filename;
    string image_path;
    float a,b;
    int c=0;
    while (getline(file, filename)) {
       // cout << "Filename: " << folder_path+"/"+filename << endl;
       image_path=folder_path+"/"+filename;

        a=actual(image_path);
        b=predict_folder(image_path,image,conv1_output,pool1_output,conv2_output,pool2_output,fc1_output,fc2_output,softmax_output
       ,conv1_kernels,conv1_biases,conv2_kernels,conv2_biases,fc1_kernels,fc1_biases,fc2_kernels,fc2_biases);

        if(a==b){
             correct++;
       }
       total++;
       c=total;
       if(c%1000==0){
         cout<<"total: "<<total <<"Accuracy: "<<(correct/total)*100.0  <<endl;
         
       }
       if(c==1000){
            break;
       }
       
    }

    file.close();
    float acc= (correct/total)*100.0 ;
    cout<<"total: "<<total <<endl;
    return acc;
}

int main() {

    auto start = chrono::high_resolution_clock::now();

    string folder_path="pre-proc-img";
    string folder_text="precessed_test_neg.txt";
    cout<<accuracy(folder_path,folder_text)<<endl;

    // Get the current time after calling the function
    auto end = chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    // Output the duration in microseconds
    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;


    //---------------------------------
    // string image_path="processed_test_case/3.txt";

    // cout<<predict(image_path) <<" " << actual(image_path)<<endl;
    
    return 0;
}
