#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <algorithm>

using namespace std;

#define EPOCHS 1000 //number of training epochs
#define LR 0.05f //learning rate
#define N_EXECUTION 5 //number of execution performed each with different kernel configuration
#define BATCH_SIZE 256 //size of each batch

#define CUDA_CHECK(call) do {                              \
  cudaError_t _err = (call);                               \
  if (_err != cudaSuccess) {                               \
    fprintf(stderr, "CUDA error %s:%d: %s\n",              \
            __FILE__, __LINE__, cudaGetErrorString(_err)); \
    exit(1);                                               \
  }                                                        \
} while(0)

//binary file structure
struct Header { uint32_t n_samples; uint32_t n_features; };
// Struct containing configurations and metrics of each execution
struct Performance{
    int numThreads;
    int numBlocks;
    float time;
    float accuracy;
    float bias;
    vector<float> weights;
};

// Kernel: each block computes a mini-batch
__global__ void logistic_regression_parallel_batches(
    float* X,
    float* y,
    float* w,
    float* b,
    float* grad_w_global,
    float* grad_b_global,
    int N,
    int F,
    int batch_size
){
    extern __shared__ float shared[];
    float* s_grad_w = shared;
    float* s_grad_b = &shared[F];

    int tid = threadIdx.x; // Thread identifier inside a block
    int batch_idx = blockIdx.x; // each block = 1 mini-batch
    int batch_start = batch_idx * batch_size;
    int local_threads = blockDim.x;
    int batch_actual_size = min(batch_size, N - batch_start);

    // initializing shared memory
    for(int j = tid; j < F; j += local_threads) s_grad_w[j] = 0.0f;
    if(tid == 0) *s_grad_b = 0.0f;

    __syncthreads();

    // computing gradients for mini-batch samples
    for(int i = tid; i < batch_actual_size; i += local_threads){
        int idx = batch_start + i;
        float z = *b; //adding bias
        //compute linear combination of features and weights
        for(int j = 0; j < F; j++)
            z += w[j] * X[idx*F + j];

        //sigmoid function for prediction
        float y_pred = 1.0f / (1.0f + expf(-z));
        // prediction error
        float error = y_pred - y[idx];
        
        // weights update 
        for(int j = 0; j < F; j++)
            atomicAdd(&s_grad_w[j], error * X[idx*F + j]);
        
        atomicAdd(s_grad_b, error); //bias update
    }

    __syncthreads();

    // updating global memory
    for(int j = tid; j < F; j += local_threads)
        atomicAdd(&grad_w_global[j], s_grad_w[j]);
    if(tid == 0)
        atomicAdd(grad_b_global, *s_grad_b);
}

// Function to save into CSV file
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

    // --- Load dataset ---
    ifstream dataset_bin("dataset_normalized.bin", ios::binary);
    if(!dataset_bin){ cerr << "Failed to open dataset_normalized.bin\n"; return 1; }
    dataset_bin.read((char*)&h, sizeof(h));
    int N = h.n_samples; //fetching number of samples
    int F = h.n_features; //fetching number of features

    vector<float> mean(F), stddev(F), X(N*F), y(N);
    //fetching data from binary file to populate structures
    dataset_bin.read((char*)mean.data(), F*sizeof(float));
    dataset_bin.read((char*)stddev.data(), F*sizeof(float));
    dataset_bin.read((char*)X.data(), N*F*sizeof(float));
    dataset_bin.read((char*)y.data(), N*sizeof(float));
    dataset_bin.close(); /close

    // GPU memory allocation ---
    CUDA_CHECK(cudaMalloc(&dX, N*F*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw, F*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_w, F*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_b, sizeof(float)));

    // data transfer from Host to GPU
    CUDA_CHECK(cudaMemcpy(dX, X.data(), N*F*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, y.data(), N*sizeof(float), cudaMemcpyHostToDevice));

    size_t sharedMemSize = F*sizeof(float) + sizeof(float); // grad_w + grad_b

    cudaEvent_t start, stop;
    // creating CUDA events used to measure execution time
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("start training...\n");
    // testing multiple kernel configuration (threads / block)
    for(int t = 0; t < N_EXECUTION; t++){
        b = 0.0f;
        vector<float> w(F, 0.1f);
        CUDA_CHECK(cudaMemcpy(dw, w.data(), F*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, &b, sizeof(float), cudaMemcpyHostToDevice));

        int threads = thread_configs[t]; //number of threads per block
        int num_block = (N + BATCH_SIZE - 1) / BATCH_SIZE; //computing number of batch

        CUDA_CHECK(cudaEventRecord(start)); //starting GPU timer
        //training loop
        for(int epoch = 0; epoch < EPOCHS; epoch++){
            //resetting gradients buffered on GPU
            CUDA_CHECK(cudaMemset(d_grad_w, 0, F*sizeof(float)));
            CUDA_CHECK(cudaMemset(d_grad_b, 0, sizeof(float)));

            // Launching kernel: each block computes one mini-batch
            logistic_regression_parallel_batches<<<num_block, threads, sharedMemSize>>>(
                dX, dy, dw, db, d_grad_w, d_grad_b, N, F, BATCH_SIZE
            );
            CUDA_CHECK(cudaDeviceSynchronize());

            // copying gradients to host
            vector<float> grad_w(F);
            float grad_b;
            CUDA_CHECK(cudaMemcpy(grad_w.data(), d_grad_w, F*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&grad_b, d_grad_b, sizeof(float), cudaMemcpyDeviceToHost));

            // updating host weights
            for(int j = 0; j < F; j++) w[j] -= LR * (grad_w[j] / N);
            b -= LR * (grad_b / N);
            
            //copying back to GPU update weights and bias
            CUDA_CHECK(cudaMemcpy(dw, w.data(), F*sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(db, &b, sizeof(float), cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaEventRecord(stop));  //stop GPU timer
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms;
        //computing elapsed time in milliseconds
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        //saving model performances
        modelPerformance[t].time = elapsed_ms;
        modelPerformance[t].numBlocks = num_block;
        modelPerformance[t].numThreads = threads;
        modelPerformance[t].weights = w;
        modelPerformance[t].bias = b;

        printf("Threads: %d  Blocks: %d  Time: %.3f ms\n",threads, num_block, elapsed_ms);
    }

    // Evaluation of each trained model
    for(int t = 0; t<N_EXECUTION; t++){
        int correct = 0;
        for(int i = 0; i < N; i++){
            float z = modelPerformance[t].bias;
            for(int j = 0; j < F; j++)
                z += X[i*F + j] * modelPerformance[t].weights[j];
            float y_hat = 1.0f / (1.0f + exp(-z));
            int prediction = (y_hat >= 0.5f) ? 1 : 0;
            //checking if prediction matches ground truth
            if(prediction == (int)y[i]) correct++;
        }
        modelPerformance[t].accuracy = (float)correct / N;
        printf("\nAccuracy: %.2f%%\n", modelPerformance[t].accuracy*100);
    }
    //saving performances on CSV file
    saveToCSV(modelPerformance, N_EXECUTION, F);

    // GPU memory cleanup
    CUDA_CHECK(cudaFree(dX)); 
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(dw)); 
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(d_grad_w));
    CUDA_CHECK(cudaFree(d_grad_b));
    //destroying cuda timing events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}