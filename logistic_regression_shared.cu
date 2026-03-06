#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

using namespace std;

#define EPOCHS 1000 //number of training epochs 
#define LR 0.05f //learning rate
#define N_EXECUTION 5 //number of execution performed each with different kernel configuration

// Binary file structure
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

// Kernel function performing logistic regression for each sample
__global__ void logistic_regression_kernel_tiles(
    float* X,
    float* y,
    float* w,
    float* b,
    float* grad_b,
    float* grad_w,
    int N,
    int F)
{
    extern __shared__ float shared[];
    float* s_grad_w = shared;          // F float per blocco
    float* s_grad_b = &shared[F];      // 1 float per blocco

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // --- inizializzazione shared memory ---
    for(int j = tid; j < F; j += blockDim.x)
        s_grad_w[j] = 0.0f;

    if(tid == 0) *s_grad_b = 0.0f;

    __syncthreads();

    // --- calcolo del gradiente per ogni thread (sample) ---
    if(i < N){
        float z = *b;
        for(int j = 0; j < F; j++)
            z += w[j] * X[i*F + j];

        float y_pred = 1.0f / (1.0f + expf(-z));
        float error = y_pred - y[i];

        for(int j = 0; j < F; j++)
            atomicAdd(&s_grad_w[j], error * X[i*F + j]);

        atomicAdd(s_grad_b, error);
    }

    __syncthreads();

    // --- un solo thread per blocco aggiorna memoria globale ---
    for(int j = tid; j < F; j += blockDim.x)
        atomicAdd(&grad_w[j], s_grad_w[j]);

    if(tid == 0)
        atomicAdd(grad_b, *s_grad_b);
}

void saveToCSV(Performance p[], int n) {

    ofstream file("modelPerformanceShared.csv");

    // intestazioni
    file << "numThreads;numBlocks;time;accuracy";
    for(int i = 0; i < 22; i++)
        file << ";weight" << i;
    file << "\n";

    // righe dati
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

// --- Main --- 
int main(){
    Header h;
    Performance modelPerformance[N_EXECUTION];
    float b = 0.0f;
    float *dX, *dy, *dw, *db, *d_grad_w, *d_grad_b;
    int thread_configs[] = {64, 128, 256, 512, 1024};
    

    ifstream dataset_bin("dataset_normalized.bin", ios::binary);
    if(!dataset_bin){
        cerr << "Failed to open dataset_normalized.bin\n";
        return 1;
    }

    dataset_bin.read((char*)&h, sizeof(h));
    int N = h.n_samples;
    int F = h.n_features;

    vector<float> mean(F), stddev(F), X(N*F), y(N);
    dataset_bin.read((char*)mean.data(), F*sizeof(float));
    dataset_bin.read((char*)stddev.data(), F*sizeof(float));
    dataset_bin.read((char*)X.data(), N*F*sizeof(float));
    dataset_bin.read((char*)y.data(), N*sizeof(float));
    dataset_bin.close();

    // --- GPU allocation ---
    cudaMalloc(&dX, N*F*sizeof(float));
    cudaMalloc(&dy, N*sizeof(float));
    cudaMalloc(&dw, F*sizeof(float));
    cudaMalloc(&db, sizeof(float));
    cudaMalloc(&d_grad_w, F*sizeof(float));
    cudaMalloc(&d_grad_b, sizeof(float));

    cudaMemcpy(dX, X.data(), N*F*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, &b, sizeof(float), cudaMemcpyHostToDevice);

    vector<float> w(F, 0.1f);
    cudaMemcpy(dw, w.data(), F*sizeof(float), cudaMemcpyHostToDevice);

    
    size_t sharedMemSize = F*sizeof(float) + sizeof(float); // grad_w + grad_b

    cout << "Training started...\n";

    // --- CUDA events per il timing ---
    cudaEvent_t start, stop;
    float elapsed_ms = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

     // inizio timing
    for(int t = 0 ; t<N_EXECUTION; t++){
        int threads = thread_configs[t];
        int blocks = (N + threads - 1) / threads;
        cudaEventRecord(start); 
        for(int epoch = 0; epoch < EPOCHS; epoch++){
            cudaMemset(d_grad_w, 0, F*sizeof(float));
            cudaMemset(d_grad_b, 0, sizeof(float));

            logistic_regression_kernel_tiles<<<blocks, threads, sharedMemSize>>>(
                dX, dy, dw, db, d_grad_b, d_grad_w, N, F
            );
            cudaDeviceSynchronize();

            // copia gradienti su host
            vector<float> grad_w(F);
            float grad_b;
            cudaMemcpy(grad_w.data(), d_grad_w, F*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&grad_b, d_grad_b, sizeof(float), cudaMemcpyDeviceToHost);

            // aggiornamento pesi batch per epoca
            for(int j = 0; j < F; j++)
                w[j] -= LR * (grad_w[j] / N);
            b -= LR * (grad_b / N);

            cudaMemcpy(dw, w.data(), F*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(db, &b, sizeof(float), cudaMemcpyHostToDevice);
        }

        cudaEventRecord(stop);  // fine timing
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_ms, start, stop);

        
        cudaMemcpy(w.data(), dw, F*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&b, db, sizeof(float), cudaMemcpyDeviceToHost);
        modelPerformance[t].time = elapsed_ms;
        modelPerformance[t].numBlocks = blocks;
        modelPerformance[t].numThreads = threads;
        modelPerformance[t].weights = w;
        printf("Threads: %d  Blocks: %d  Time: %.3f ms\n",threads, blocks, elapsed_ms);
    }
    

    for(int t = 0; t<N_EXECUTION; t++){
        int correct = 0;
        for(int i = 0; i < N; i++){
            float z = b;
            for(int j = 0; j < F; j++)
               // z += X[i*F + j] * w[j];
                z += X[i*F + j] * modelPerformance[t].weights[j];

            float y_hat = 1.0f / (1.0f + exp(-z));

            int prediction = (y_hat >= 0.5f) ? 1 : 0;

            if(prediction == (int)y[i])
                correct++;
        }

        float accuracy = (float)correct / N;
        modelPerformance[t].accuracy = accuracy;
        printf("\nAccuracy: %.2f%%\n", accuracy * 100.0f);
    }

    saveToCSV(modelPerformance, N_EXECUTION);

    cudaFree(dX); cudaFree(dy); cudaFree(dw); cudaFree(db);
    cudaFree(d_grad_w); cudaFree(d_grad_b);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}