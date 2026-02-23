#include <fstream>
#include <vector>
#include <cuda_runtime.h>
using namespace std;

struct Header {
    uint32_t n_samples;
    uint32_t n_features;
};

__global__ void regression_kernel(float* X, float* y, float* weights, int n_samples, int n_features) {
}

int main() {
    //load dataset from binary file generated before
    ifstream dataset_bin("dataset.bin", ios::binary);
    if(!dataset_bin) {
        fprintf(stderr, "Failed to open dataset.bin\n");
        return 1;
    }

    Header h;
    dataset_bin.read((char*)&h, sizeof(h));

    //calculate the size of X that are the number of features
    size_t X_size = h.n_samples * h.n_features;
    //calculate the size of y that are the number of true labels
    size_t y_size = h.n_samples;

    //creating vectors to hold the data
    vector<float> X(X_size);
    vector<float> y(y_size);

    //read the values from the dataset.bin file into the vectors
    dataset_bin.read((char*)X.data(), X_size * sizeof(float));
    dataset_bin.read((char*)y.data(), y_size * sizeof(float));
    

    dataset_bin.close();

    // GPU allocation
    float *dX, *dy, *dw, *dy_hat;
    cudaMalloc(&dX, X_size * sizeof(float));
    cudaMalloc(&dy, y_size * sizeof(float));
    cudaMalloc(&dw, h.n_features * sizeof(float));
    cudaMalloc(&dy_hat, y_size * sizeof(float));

    //copy data from host to GPU
    cudaMemcpy(dX, X.data(), X_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.data(), y_size * sizeof(float), cudaMemcpyHostToDevice);
    

    printf("Dataset loaded on GPU: %d samples, %d features\n",
           h.n_samples, h.n_features);
           
    
}