#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <algorithm>

using namespace std;

#define EPOCHS 1000
#define LR 0.05f
#define N_EXECUTION 5
#define BATCH_SIZE 2048

#define CUDA_CHECK(call) do {                              \
  cudaError_t _err = (call);                               \
  if (_err != cudaSuccess) {                               \
    fprintf(stderr, "CUDA error %s:%d: %s\n",              \
            __FILE__, __LINE__, cudaGetErrorString(_err)); \
    exit(1);                                               \
  }                                                        \
} while(0)

struct Header { uint32_t n_samples; uint32_t n_features; };
struct Performance{
    int numThreads;
    int numBlocks;
    float time;
    float accuracy;
    float bias;
    vector<float> weights;
};

// Kernel per mini-batch sequenziale
__global__ void logistic_regression_kernel_sequential(
    float* X,
    float* y,
    float* w,
    float* b,
    float* grad_b,
    float* grad_w,
    int batch_start,
    int batch_size,
    int F
){
    extern __shared__ float shared[];
    float* s_grad_w = shared;
    float* s_grad_b = &shared[F];

    int tid = threadIdx.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    for(int j = tid; j < F; j += blockDim.x) s_grad_w[j] = 0.0f;
    if(tid == 0) *s_grad_b = 0.0f;

    __syncthreads();

    if(i < batch_size){
        int idx = batch_start + i;
        float z = *b;
        for(int j = 0; j < F; j++)
            z += w[j] * X[idx*F + j];

        float y_pred = 1.0f / (1.0f + expf(-z));
        float error = y_pred - y[idx];

        for(int j = 0; j < F; j++)
            atomicAdd(&s_grad_w[j], error * X[idx*F + j]);
        atomicAdd(s_grad_b, error);
    }

    __syncthreads();

    for(int j = tid; j < F; j += blockDim.x)
        atomicAdd(&grad_w[j], s_grad_w[j]);
    if(tid == 0)
        atomicAdd(grad_b, *s_grad_b);
}

void saveToCSV(Performance p[], int n, int f) {
    ofstream file("modelPerformanceMinibatch.csv");
    file << "numThreads;numBlocks;time;accuracy";
    for(int i = 0; i < f; i++) file << ";weight" << i;
    file << "\n";

    for(int i = 0; i < n; i++) {
        file << p[i].numThreads << ";"
             << p[i].numBlocks << ";"
             << p[i].time << ";"
             << p[i].accuracy;
        for(int j = 0; j < f; j++) file << ";" << p[i].weights[j];
        file << "\n";
    }
}

int main(){
    Header h;
    Performance modelPerformance[N_EXECUTION];
    float b = 0.0f;
    float *dX, *dy, *dw, *db, *d_grad_w, *d_grad_b;
    int thread_configs[] = {64,128,256,512,1024};

    ifstream dataset_bin("dataset_normalized.bin", ios::binary);
    if(!dataset_bin){ cerr << "Failed to open dataset_normalized.bin\n"; return 1; }
    dataset_bin.read((char*)&h, sizeof(h));
    int N = h.n_samples;
    int F = h.n_features;

    vector<float> mean(F), stddev(F), X(N*F), y(N);
    dataset_bin.read((char*)mean.data(), F*sizeof(float));
    dataset_bin.read((char*)stddev.data(), F*sizeof(float));
    dataset_bin.read((char*)X.data(), N*F*sizeof(float));
    dataset_bin.read((char*)y.data(), N*sizeof(float));
    dataset_bin.close();

    CUDA_CHECK(cudaMalloc(&dX, N*F*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw, F*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_w, F*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_b, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dX, X.data(), N*F*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, y.data(), N*sizeof(float), cudaMemcpyHostToDevice));

    size_t sharedMemSize = F*sizeof(float) + sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("start training...\n");
    for(int t = 0; t < N_EXECUTION; t++){
        b = 0.0f;
        vector<float> w(F, 0.1f);
        CUDA_CHECK(cudaMemcpy(dw, w.data(), F*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, &b, sizeof(float), cudaMemcpyHostToDevice));

        int threads = thread_configs[t];
        int blocks = (BATCH_SIZE + threads - 1) / threads;
        
        CUDA_CHECK(cudaEventRecord(start));

        for(int epoch = 0; epoch < EPOCHS; epoch++){
            for(int batch_start = 0; batch_start < N; batch_start += BATCH_SIZE){
                int batch_size = min(BATCH_SIZE, N - batch_start);

                CUDA_CHECK(cudaMemset(d_grad_w, 0, F*sizeof(float)));
                CUDA_CHECK(cudaMemset(d_grad_b, 0, sizeof(float)));

                logistic_regression_kernel_sequential<<<blocks, threads, sharedMemSize>>>(
                    dX, dy, dw, db, d_grad_b, d_grad_w, batch_start, batch_size, F
                );
                CUDA_CHECK(cudaDeviceSynchronize());

                vector<float> grad_w(F);
                float grad_b;
                CUDA_CHECK(cudaMemcpy(grad_w.data(), d_grad_w, F*sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&grad_b, d_grad_b, sizeof(float), cudaMemcpyDeviceToHost));

                for(int j = 0; j < F; j++) w[j] -= LR * (grad_w[j] / batch_size);
                b -= LR * (grad_b / batch_size);

                CUDA_CHECK(cudaMemcpy(dw, w.data(), F*sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(db, &b, sizeof(float), cudaMemcpyHostToDevice));
            }
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        modelPerformance[t].time = elapsed_ms;
        modelPerformance[t].numBlocks = blocks;
        modelPerformance[t].numThreads = threads;
        modelPerformance[t].weights = w;
        modelPerformance[t].bias = b;

        printf("Threads: %d  Blocks: %d  Time: %.3f ms\n",threads, blocks, elapsed_ms);
    }

    for(int t = 0; t<N_EXECUTION; t++){
        int correct = 0;
        for(int i = 0; i < N; i++){
            float z = modelPerformance[t].bias;
            for(int j = 0; j < F; j++)
                z += X[i*F + j] * modelPerformance[t].weights[j];
            float y_hat = 1.0f / (1.0f + exp(-z));
            int prediction = (y_hat >= 0.5f) ? 1 : 0;
            if(prediction == (int)y[i]) correct++;
        }
        modelPerformance[t].accuracy = (float)correct / N;
        printf("\nAccuracy: %.2f%%\n", modelPerformance[t].accuracy*100);
    }

    saveToCSV(modelPerformance, N_EXECUTION, F);

    CUDA_CHECK(cudaFree(dX)); 
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(dw)); 
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(d_grad_w));
    CUDA_CHECK(cudaFree(d_grad_b));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}