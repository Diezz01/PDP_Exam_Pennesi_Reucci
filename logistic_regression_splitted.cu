#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
using namespace std;

#define EPOCHS 1000 //number of training epochs
#define LR 0.05f //learning rate
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
    vector<float> weights;
};

// Kernel 1: compute linear combination z = w*x + b
__global__ void linear_kernel(
    float* X,
    float* w,
    float* b,
    float* z_out,
    int N,
    int F)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){
        float z = 0.0f;

        //compute linear combination of features and weights
        for(int j = 0; j < F; j++){
            z += X[i*F + j] * w[j];
        }

        //adding bias
        z += *b;

        //store result for next kernel
        z_out[i] = z;
    }
}

// Kernel 2: compute sigmoid and prediction error
__global__ void sigmoid_error_kernel(
    float* z,
    float* y,
    float* error,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){

        //sigmoid function
        float y_pred = 1.0f / (1.0f + expf(-z[i]));

        //error of prediction
        error[i] = y_pred - y[i];
    }
}

// Kernel 3: compute gradients
__global__ void gradient_kernel(
    float* X,
    float* error,
    float* grad_w,
    float* grad_b,
    int N,
    int F)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){

        // weights gradient update
        for(int j = 0; j < F; j++){
            atomicAdd(&grad_w[j], error[i] * X[i*F + j]);
        }

        //bias gradient update
        atomicAdd(grad_b, error[i]);
    }
}

//function used to save perfromances in a csv file 
void saveToCSV(Performance p[], int n) {

    ofstream file("modelPerformanceSplitted.csv");

    // header
    file << "numThreads;numBlocks;time;accuracy";
    for(int i = 0; i < 22; i++)
        file << ";weight" << i;
    file << "\n";

    // performances data
    for(int i = 0; i < n; i++) {

        file << p[i].numThreads << ";"
             << p[i].numBlocks << ";"
             << p[i].time << ";"
             << p[i].accuracy;

        for(int j = 0; j < 22; j++)
            file << ";" << p[i].weights[j];

        file << "\n";
    }
}

int main(){

    Header h;
    Performance modelPerformance[N_EXECUTION];

    float b = 0.0f; //bias of the model

    float *dX;//cuda list of features on GPU
    float *dy;//cuda array of true label on GPU
    float *dw;//cuda array of features weights on GPU
    float *db;//cuda model bias on GPU

    float *d_grad_w;//cuda arry of weights gradient calculated in the kernel
    float *d_grad_b;//cuda gradient of the bias calculated in the kernel

    float *dz;        // cuda linear output
    float *derror;    // cuda prediction error


    //different threads to launch different configuration
    int thread_configs[] = {64, 128, 256, 512, 1024};

    //cuda event to calculate training execution time
    cudaEvent_t start, stop;
    float ms = 0;//execution time in millisecons

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
    vector<float> X(N * F); //matrix of features (N x F)
    vector<float> y(N); //true labl vector (0 or 1)
    vector<float> w(F, 0.1f);//initialization of weights array

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

    CUDA_CHECK(cudaMalloc(&dz, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&derror, N * sizeof(float)));

    // data transfer from Host to GPU
    CUDA_CHECK(cudaMemcpy(dw, w.data(), F * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dX, X.data(), X_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, y.data(), y_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, &b, sizeof(float), cudaMemcpyHostToDevice));

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
    for(int t = 0 ; t<N_EXECUTION; t++){

        int threads = thread_configs[t];//number of threads per block
        int blocks = (N + threads - 1) / threads; //compute number of blocks needed

        CUDA_CHECK(cudaEventRecord(start));//starting GPU timer

        //training loop
        for(int epoch = 0; epoch<EPOCHS; epoch++){

            //resetting gradients buffered on GPU
            CUDA_CHECK(cudaMemset(d_grad_w, 0, F * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_grad_b, 0, sizeof(float)));

            // compute linear combination
            linear_kernel<<<blocks, threads>>>(dX, dw, db, dz, N, F);

            // compute sigmoid and error
            sigmoid_error_kernel<<<blocks, threads>>>(dz, dy, derror, N);

            // compute gradients
            gradient_kernel<<<blocks, threads>>>(dX, derror, d_grad_w, d_grad_b, N, F);

            // Waiting for kernel execution to finish
            CUDA_CHECK(cudaDeviceSynchronize());

            vector<float> grad_w(F);//host buffer for weight gradients
            float grad_b;//host buffer for bias gradient

            CUDA_CHECK(cudaMemcpy(grad_w.data(), d_grad_w, F*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&grad_b, d_grad_b, sizeof(float), cudaMemcpyDeviceToHost));

            //updating weights using gradient descent
            for(int j = 0; j < F; j++)
                w[j] -= LR * (grad_w[j] / N);

            b -= LR * (grad_b / N);
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

        printf("Threads: %d  Blocks: %d  Time: %.3f ms\n",threads, blocks, ms);
    }

    //evaluation of each trained model
    for(int t = 0; t<N_EXECUTION; t++){

        int correct = 0;

        for(int i = 0; i < N; i++){

            float z = b;
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
    saveToCSV(modelPerformance, N_EXECUTION);
    //GPU memory cleanup
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(dw));
    CUDA_CHECK(cudaFree(db));

    CUDA_CHECK(cudaFree(dz));
    CUDA_CHECK(cudaFree(derror));

    //destroying cuda timing events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}