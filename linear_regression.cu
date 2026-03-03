#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

using namespace std;

#define F_FIXED 10   // sappiamo che F = 10

struct Header {
    uint32_t n_samples;
    uint32_t n_features;
};

__global__ void regression_epoch(
    float* X,
    float* y,
    float* w,
    float b,
    float* mse,
    float* grad_w,
    float* grad_b,
    int N)
{
    __shared__ float s_grad_w[F_FIXED];
    __shared__ float s_grad_b;
    __shared__ float s_mse;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // init shared memory
    if (tid < F_FIXED)
        s_grad_w[tid] = 0.0f;

    if (tid == 0) {
        s_grad_b = 0.0f;
        s_mse = 0.0f;
    }

    __syncthreads();

    if (i < N) {

        float pred = b;

        for (int j = 0; j < F_FIXED; j++)
            pred += X[i * F_FIXED + j] * w[j];

        float error = pred - y[i];

        atomicAdd(&s_mse, error * error);
        atomicAdd(&s_grad_b, error);

        for (int j = 0; j < F_FIXED; j++)
            atomicAdd(&s_grad_w[j], error * X[i * F_FIXED + j]);
    }

    __syncthreads();

    // write block results to global
    if (tid == 0) {
        atomicAdd(mse, s_mse);
        atomicAdd(grad_b, s_grad_b);

        for (int j = 0; j < F_FIXED; j++)
            atomicAdd(&grad_w[j], s_grad_w[j]);
    }
}

int main() {

    Header h;
    float b = 0.0f;

    float *dX, *dy, *dw;
    float *d_mse, *d_grad_w, *d_grad_b;

    ifstream dataset_bin("dataset.bin", ios::binary);
    if (!dataset_bin) {
        fprintf(stderr, "Failed to open dataset.bin\n");
        return 1;
    }

    dataset_bin.read((char*)&h, sizeof(h));

    int N = h.n_samples;
    int F = h.n_features;

    if (F != F_FIXED) {
        cout << "Questo codice è ottimizzato per F=10\n";
        return 1;
    }

    size_t X_size = N * F;
    size_t y_size = N;

    vector<float> X(X_size);
    vector<float> y(y_size);

    dataset_bin.read((char*)X.data(), X_size * sizeof(float));
    dataset_bin.read((char*)y.data(), y_size * sizeof(float));
    dataset_bin.close();

    // GPU allocation
    cudaMalloc(&dX, X_size * sizeof(float));
    cudaMalloc(&dy, y_size * sizeof(float));
    cudaMalloc(&dw, F * sizeof(float));

    cudaMalloc(&d_mse, sizeof(float));
    cudaMalloc(&d_grad_w, F * sizeof(float));
    cudaMalloc(&d_grad_b, sizeof(float));

    // copy dataset once
    cudaMemcpy(dX, X.data(), X_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.data(), y_size * sizeof(float), cudaMemcpyHostToDevice);

    vector<float> w(F, 0.1f);
    cudaMemcpy(dw, w.data(), F * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float lr = 1e-8f;
    int epochs = 100;

    cout << "Training started...\n";

    for (int e = 0; e < epochs; e++) {

        cudaMemset(d_mse, 0, sizeof(float));
        cudaMemset(d_grad_w, 0, F * sizeof(float));
        cudaMemset(d_grad_b, 0, sizeof(float));

        regression_epoch<<<blocks, threads>>>(
            dX, dy, dw, b,
            d_mse, d_grad_w, d_grad_b,
            N);

        cudaDeviceSynchronize();

        float mse, grad_b;
        vector<float> grad_w(F);

        cudaMemcpy(&mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&grad_b, d_grad_b, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_w.data(), d_grad_w, F * sizeof(float), cudaMemcpyDeviceToHost);

        mse /= N;
        grad_b = (2.0f / N) * grad_b;

        for (int j = 0; j < F; j++)
            grad_w[j] = (2.0f / N) * grad_w[j];

        b -= lr * grad_b;

        for (int j = 0; j < F; j++)
            w[j] -= lr * grad_w[j];

        cudaMemcpy(dw, w.data(), F * sizeof(float), cudaMemcpyHostToDevice);
        
        cout << "Epoch " << e << " | MSE: " << mse << endl;
        cout <<"-------------------------------------------" << endl;
    }

    cout << "\nTraining completed.\n";
    cout << "Final bias: " << b << endl;

    cout << "Final weights:\n";
    for (int j = 0; j < F; j++)
        cout << w[j] << " ";

    cout << endl;

    cudaFree(dX);
    cudaFree(dy);
    cudaFree(dw);
    cudaFree(d_mse);
    cudaFree(d_grad_w);
    cudaFree(d_grad_b);

    return 0;
}