#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
using namespace std;

struct Header {
    uint32_t n_samples;
    uint32_t n_features;
};

__global__ void regression_kernel(
    float* X,
    float* y,
    float* w,
    float b,
    float* y_hat_out,
    float* mse,
    float* grad_w,
    float* grad_b,
    int N,
    int F)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float y_hat = b;

        for (int j = 0; j < F; j++)
            y_hat += X[i * F + j] * w[j];

        y_hat_out[i] = y_hat;

        float error = y_hat - y[i];

        // Accumulating Mean Squared Error (MSE)
        atomicAdd(mse, error * error);

        // Gradiente bias
        atomicAdd(grad_b, error);

        // Gradiente pesi
        for (int j = 0; j < F; j++)
            atomicAdd(&grad_w[j], error * X[i * F + j]);
    }
}

int main() {
    Header h;
    float b = 0.0f; // Bias
    float *dX, *dy, *dw, *dy_hat;        // GPU pointers for input, weights, bias, and predictions
    float *d_mse, *d_grad_w, *d_grad_b; // GPU pointers for MSE and gradients
    
    // Loading dataset from binary file generated before
    ifstream dataset_bin("/content/drive/MyDrive/dataset.bin", ios::binary);
    if(!dataset_bin) {
        fprintf(stderr, "Failed to open dataset.bin\n");
        return 1;
    }

    // Reading the header from the dataset.bin file
    dataset_bin.read((char*)&h, sizeof(h));

    int N = h.n_samples;   // Number of samples
    int F = h.n_features; // Number of features
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t X_size = h.n_samples * h.n_features; // Calculating the size of X that are the number of FEATURES
    size_t y_size = h.n_samples;               // Calculating the size of y that are the number of TRUE LABELS
    float mse, grad_b, lr = 1e-5;
    vector<float> grad_w(F);

    // Creating vectors to hold X and y
    vector<float> X(X_size);
    vector<float> y(y_size);

    // Reading values from the dataset.bin file into the vectors
    dataset_bin.read((char*)X.data(), X_size * sizeof(float));
    dataset_bin.read((char*)y.data(), y_size * sizeof(float));

    dataset_bin.close();

    // GPU allocation
    cudaMalloc(&dX, X_size * sizeof(float));
    cudaMalloc(&dy, y_size * sizeof(float));
    cudaMalloc(&dw, h.n_features * sizeof(float));
    cudaMalloc(&dy_hat, y_size * sizeof(float));

    // GPU allocation for MSE and gradients
    cudaMalloc(&d_mse, sizeof(float));
    cudaMalloc(&d_grad_w, F * sizeof(float));
    cudaMalloc(&d_grad_b, sizeof(float));

    // Setting MSE and gradients values to zero before kernel
    cudaMemset(d_mse, 0, sizeof(float));
    cudaMemset(d_grad_w, 0, F * sizeof(float));
    cudaMemset(d_grad_b, 0, sizeof(float));

    printf("Dataset loaded on GPU: %d samples, %d features\n",
           h.n_samples, h.n_features);

    // Setting all the weigths to 0.1
    vector<float> w(F, 0.1f);

    // Copying data from host to GPU
    cudaMemcpy(dw, w.data(), F * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X.data(), X_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.data(), y_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launching kernel
    regression_kernel<<<blocks, threads>>>(
    dX, dy, dw, b,
    dy_hat, d_mse, d_grad_w, d_grad_b,
    N, F);

    cudaDeviceSynchronize();

    vector<float> y_hat(10);
    cudaMemcpy(y_hat.data(), dy_hat, 10 * sizeof(float), cudaMemcpyDeviceToHost);

    // Copying back MSE and gradients values from GPU to host
    cudaMemcpy(&mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&grad_b, d_grad_b, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_w.data(), d_grad_w, F * sizeof(float), cudaMemcpyDeviceToHost);

    mse /= N; // Dividing MSE by the number of samples

    grad_b = (2.0f / N) * grad_b; // Dividing gradient by the number of samples

    // Dividing gradient by the number of samples
    for (int j = 0; j < F; j++)
        grad_w[j] = (2.0f / N) * grad_w[j];

    // Updating bias
    b -= lr * grad_b;

    // Updating weights
    for (int j = 0; j < F; j++)
        w[j] -= lr * grad_w[j];

    cudaMemcpy(dw, w.data(), F * sizeof(float), cudaMemcpyHostToDevice); // Copying updated weights back to GPU

    cout << "MSE: " << mse << endl;
    cout << "Grad_b: " << grad_b << endl;

    cout << "\nSample predictions:\n";
    for (int i = 0; i < 10; i++)
        cout << "y_hat[" << i << "] = " << y_hat[i] << "\n";

    cudaFree(dX);
    cudaFree(dy);
    cudaFree(dw);
    cudaFree(dy_hat);

    cout << "\nGPU forward pass SUCCESS\n";
    return 0;
}