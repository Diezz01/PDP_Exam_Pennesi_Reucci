#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
using namespace std;

#define EPOCHS 1000//number of training epochs
#define LR 0.05f//learning rate
#define N_EXECUTION 5 //number of execution performed each with different kernel configuration

#define CUDA_CHECK(call) do {                              \
  cudaError_t _err = (call);                               \
  if (_err != cudaSuccess) {                               \
    fprintf(stderr, "CUDA error %s:%d: %s\n",              \
            __FILE__, __LINE__, cudaGetErrorString(_err)); \
    exit(1);                                               \
  }                                                        \
} while(0)

//binary file structure
struct Header {
    uint32_t n_samples;
    uint32_t n_features;
};

// Struct containing configurations and metrics of each execution
struct Performance{
    int numThreads;
    int numBlocks;
    float time;
    float accuracy;
    float bias;
    vector<float> weights;
};

// Kernel function performing logistic regression for each sample
__global__ void logistic_regression_kernel(
    float* X,
    float* y,
    float* w,
    float* b,
    float* grad_b,
    float* grad_w,
    int N,
    int F)
{
    int tid = threadIdx.x; // Thread identifier inside a block
    int i = blockIdx.x * blockDim.x + tid; // Global thread identifier

    if(i < N){
        float z = 0.0f;
        float y_pred = 0;
        //compute linear combination of features and weights
        for (int j = 0; j < F; j++){
            z += X[F*i +j] * w[j];
        }
        //adding bias
        z += *b;
        //sigmoid function for prediction
        y_pred = 1.0f / (1.0f + expf(-z));

        //error of prediction
        float error = y_pred - y[i];

        // weights update 
        for(int j = 0; j < F; j++){
            atomicAdd(&grad_w[j], error * X[i*F + j]);
            
        }
        //bias update
        atomicAdd(grad_b, error);
        
    }

}

//function used to save perfromances in a csv file 
void saveToCSV(Performance p[], int n, int f) {

    ofstream file("modelPerformance.csv");

    // header
    file << "numThreads;numBlocks;time;accuracy";
    for(int i = 0; i < f; i++)
        file << ";weight" << i;
    file << "\n";
    // performances data
    for(int i = 0; i < n; i++) {

        file << p[i].numThreads << ";"
             << p[i].numBlocks << ";"
             << p[i].time << ";"
             << p[i].accuracy;

        for(int j = 0; j < f; j++)
            file << ";" << p[i].weights[j];

        file << "\n";
    }
}


int main(){
    Header h;
    Performance modelPerformance[N_EXECUTION];
    float *dX;//cuda list of features on GPU 
    float *dy;//cuda array of true label on GPU
    float *dw;//cuda array of features weights on GPU
    float *db;//cuda model bias on GPU
    float *d_grad_w;//cuda arry of weights gradient calculated in the kernel
    float *d_grad_b;//cuda gradient of the bias calculated in the kernel

    //different threads to launch different configuration
    int thread_configs[] = {64, 128, 256, 512, 1024};

    //cuda event to calculate training execution time
    cudaEvent_t start, stop;
    float ms = 0; //execution time in millisecons

    //fetching dataset from binary file
    ifstream dataset_bin("dataset_normalized.bin", ios::binary);
    if (!dataset_bin) {
        fprintf(stderr, "Failed to open dataset_normalized.bin\n");
        return 1;
    }

    dataset_bin.read((char*)&h, sizeof(h));
    int N = h.n_samples;//fetching number of samples
    int F = h.n_features;//fetching number of features
   
    size_t X_size = N * F;//cardinality matrix of data
    size_t y_size = N; //cardinality of label

    vector<float> mean(F);
    vector<float> stddev(F);
    vector<float> X(N * F);//matrix of features (N x F)
    vector<float> y(N);//true labl vector (0 or 1)

    //fetching data from binary file to populate structures
    dataset_bin.read((char*)mean.data(), F * sizeof(float));
    dataset_bin.read((char*)stddev.data(), F * sizeof(float));
    dataset_bin.read((char*)X.data(), N * F * sizeof(float));
    dataset_bin.read((char*)y.data(), N * sizeof(float));

    dataset_bin.close();//close 

    // GPU memory allocation
    CUDA_CHECK(cudaMalloc(&dX, X_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, y_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw, F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_w, F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_b, sizeof(float)));

    // data transfer from Host to GPU
    CUDA_CHECK(cudaMemcpy(dX, X.data(), X_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, y.data(), y_size * sizeof(float), cudaMemcpyHostToDevice));


    // counting how many samples belong to each class
    int count0 = 0, count1 = 0;
    for(int i = 0; i < N; i++){
        if((int)y[i] == 0) count0++;
        else count1++;
    }
    printf("Class 0: %d\n", count0);
    printf("Class 1: %d\n", count1);

    // creating CUDA events used to measure execution time
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cout << "Training started...\n";
    // testing multiple kernel configuration (threads / block)
   
    float b = 0.0f;
    for(int t = 0 ; t<N_EXECUTION; t++){
        b=0.0f;
        vector<float> w(F, 0.1f);
        CUDA_CHECK(cudaMemcpy(dw, w.data(), F*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, &b, sizeof(float), cudaMemcpyHostToDevice));
        int threads = thread_configs[t];//number of threads per block
        int blocks = (N + threads - 1) / threads;  // compute number of blocks needed

        CUDA_CHECK(cudaEventRecord(start));//starting GPU timer
        //training loop
        for(int epoch = 0; epoch<EPOCHS; epoch++){
            //resetting gradients buffered on GPU
            CUDA_CHECK(cudaMemset(d_grad_w, 0, F * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_grad_b, 0, sizeof(float)));

            // Launching logistic regression kernel
            logistic_regression_kernel<<<blocks, threads>>>(dX, dy, dw, db, d_grad_b, d_grad_w, N, F);
            // Waiting for kernel execution to finish
            CUDA_CHECK(cudaDeviceSynchronize());

            vector<float> grad_w(F); //host buffer for weight gradients
            float grad_b; //host buffer for bias gradient
            
            //retrieving gradients from GPU to host
            CUDA_CHECK(cudaMemcpy(grad_w.data(), d_grad_w, F*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&grad_b, d_grad_b, sizeof(float), cudaMemcpyDeviceToHost));

            //updating weights using gradient descent (once per epoch)
            for(int j = 0; j < F; j++)
                w[j] -= LR * (grad_w[j] / N);

            b -= LR * (grad_b / N); //bias update

            //copy back to GPU update weights and bias
            CUDA_CHECK(cudaMemcpy(dw, w.data(), F*sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(db, &b, sizeof(float), cudaMemcpyHostToDevice));
            
        }

        //stop GPU timer
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        //compute elapsed time in milliseconds
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        //retrieving final model parameters from GPU to Host
        CUDA_CHECK(cudaMemcpy(w.data(), dw, F*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&b, db, sizeof(float), cudaMemcpyDeviceToHost));

        //saving model performances
        modelPerformance[t].time = ms;
        modelPerformance[t].numBlocks = blocks;
        modelPerformance[t].numThreads = threads;
        modelPerformance[t].weights = w;
        modelPerformance[t].bias = b;
        printf("Threads: %d  Blocks: %d  Time: %.3f ms\n",threads, blocks, ms);
    }
    
    //evaluation of each trained model
    for(int t = 0; t<N_EXECUTION; t++){
        int correct = 0;
        for(int i = 0; i < N; i++){
            float z = modelPerformance[t].bias;
            //computing linear combination of features and weights
            for(int j = 0; j < F; j++)
                z += X[i*F + j] * modelPerformance[t].weights[j];
            //applying sigmoid activation
            float y_hat = 1.0f / (1.0f + exp(-z));

            int prediction = (y_hat >= 0.5f) ? 1 : 0;
            //checking if prediction matches ground truth
            if(prediction == (int)y[i])
                correct++;
        }
        //computing accuracy metrics
        float accuracy = (float)correct / N;
        modelPerformance[t].accuracy = accuracy;
        printf("\nAccuracy: %.2f%%\n", accuracy * 100.0f);
    }
    //saving performances on CSV file
    saveToCSV(modelPerformance, N_EXECUTION, F);
    
    //GPU memory cleanup
    CUDA_CHECK(cudaFree(dX)); 
    CUDA_CHECK(cudaFree(dy)); 
    CUDA_CHECK(cudaFree(dw)); 
    CUDA_CHECK(cudaFree(db));
    //destroying cuda timing events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}