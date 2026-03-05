#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

using namespace std;
#define EPOCHS 1000
#define LR 0.05f
#define THREADS 256

//binary file structure
struct Header {
    uint32_t n_samples;
    uint32_t n_features;
};

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
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid; 

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

        // weights update using SGD
        for(int j = 0; j < F; j++){
            atomicAdd(&grad_w[j], error * X[i*F + j]);
            
        }
        atomicAdd(grad_b, error);
        
    
    }

}

int main(){
    Header h;
    float b = 0.0f;
    float *dX;//cuda array of features
    float *dy;//cuda array of true label
    float *dw;//cuda array of features weights
    float *db;//cuda bias
    float *d_grad_w;
    float *d_grad_b;
    
    //SETTING TO TIME THE EXECUTION
    cudaEvent_t start, stop;
    float ms = 0; //milli seconds
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    ifstream dataset_bin("dataset_normalized.bin", ios::binary);
    if (!dataset_bin) {
        fprintf(stderr, "Failed to open dataset_normalized.bin\n");
        return 1;
    }

    dataset_bin.read((char*)&h, sizeof(h));
    int N = h.n_samples;//number of samples
    int F = h.n_features;//number of features
   
    int blocks = (N + THREADS - 1) / THREADS;

    size_t X_size = N * F;//cardinality matrix of data
    size_t y_size = N; //cardinality of label

    vector<float> mean(F);
    vector<float> stddev(F);
    vector<float> X(N * F);
    vector<float> y(N);

    dataset_bin.read((char*)mean.data(), F * sizeof(float));
    dataset_bin.read((char*)stddev.data(), F * sizeof(float));
    dataset_bin.read((char*)X.data(), N * F * sizeof(float));
    dataset_bin.read((char*)y.data(), N * sizeof(float));

    dataset_bin.close();

    // GPU allocation
    cudaMalloc(&dX, X_size * sizeof(float));
    cudaMalloc(&dy, y_size * sizeof(float));
    cudaMalloc(&dw, F * sizeof(float));
    cudaMalloc(&db, sizeof(float));
    cudaMalloc(&d_grad_w, F * sizeof(float));
    cudaMalloc(&d_grad_b, sizeof(float));

    // copy dataset
    cudaMemcpy(dX, X.data(), X_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.data(), y_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, &b, sizeof(float), cudaMemcpyHostToDevice);
    //initialization of weights array
    vector<float> w(F, 0.1f);
    cudaMemcpy(dw, w.data(), F * sizeof(float), cudaMemcpyHostToDevice);

    int count0 = 0, count1 = 0;

    for(int i = 0; i < N; i++){
        if((int)y[i] == 0) count0++;
        else count1++;
    }

    printf("Class 0: %d\n", count0);
    printf("Class 1: %d\n", count1);

    cout << "Training started...\n";
    cudaEventRecord(start);
    for(int epoch = 0; epoch<EPOCHS; epoch++){

        // reset gradienti a 0
        cudaMemset(d_grad_w, 0, F * sizeof(float));
        cudaMemset(d_grad_b, 0, sizeof(float));
        logistic_regression_kernel<<<blocks, THREADS>>>(dX, dy, dw, db, d_grad_b, d_grad_w, N, F);
        cudaDeviceSynchronize();

        // copia gradienti su host
        vector<float> grad_w(F);
        float grad_b;

        cudaMemcpy(grad_w.data(), d_grad_w, F*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&grad_b, d_grad_b, sizeof(float), cudaMemcpyDeviceToHost);

        // aggiornamento pesi (una sola volta per epoca)
        for(int j = 0; j < F; j++)
            w[j] -= LR * (grad_w[j] / N);

        b -= LR * (grad_b / N);

        // copia nuovi pesi su device
        cudaMemcpy(dw, w.data(), F*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(db, &b, sizeof(float), cudaMemcpyHostToDevice);
        
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);
    printf("Time: %f ms\n", ms);

    cudaMemcpy(w.data(), dw, F*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&b, db, sizeof(float), cudaMemcpyDeviceToHost);

    int correct = 0;

    for(int i = 0; i < N; i++){
        float z = b;

        for(int j = 0; j < F; j++)
            z += X[i*F + j] * w[j];

        float y_hat = 1.0f / (1.0f + exp(-z));

        int prediction = (y_hat >= 0.5f) ? 1 : 0;

        if(prediction == (int)y[i])
            correct++;

       // printf("Sample %d -> Pred: %.3f, Class: %d, Label: %.1f\n",
          //  i, y_hat, prediction, y[i]);
    }

    float accuracy = (float)correct / N;

    printf("\nAccuracy: %.2f%%\n", accuracy * 100.0f);
    cudaFree(dX); 
    cudaFree(dy); 
    cudaFree(dw); 
    cudaFree(db);
}