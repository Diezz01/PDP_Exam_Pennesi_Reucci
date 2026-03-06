#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std;

#define EPOCHS 500
#define LR 0.05f

struct Header {
    uint32_t n_samples;
    uint32_t n_features;
};

float sigmoid(float z){
    return 1.0f / (1.0f + exp(-z));
}

int main(){

    Header h;

    ifstream dataset_bin("dataset_normalized.bin", ios::binary);
    if(!dataset_bin){
        cout << "Error opening dataset\n";
        return 1;
    }

    dataset_bin.read((char*)&h, sizeof(h));

    int N = h.n_samples;
    int F = h.n_features;

    vector<float> mean(F);
    vector<float> stddev(F);
    vector<float> X(N * F);
    vector<float> y(N);

    dataset_bin.read((char*)mean.data(), F*sizeof(float));
    dataset_bin.read((char*)stddev.data(), F*sizeof(float));
    dataset_bin.read((char*)X.data(), N*F*sizeof(float));
    dataset_bin.read((char*)y.data(), N*sizeof(float));

    dataset_bin.close();

    vector<float> w(F, 0.1f);
    float b = 0.0f;

    cout << "Training started...\n";

    auto start = chrono::high_resolution_clock::now();

    for(int epoch = 0; epoch < EPOCHS; epoch++){

        vector<float> grad_w(F, 0.0f);
        float grad_b = 0.0f;

        for(int i = 0; i < N; i++){

            float z = b;

            for(int j = 0; j < F; j++)
                z += X[i*F + j] * w[j];

            float y_pred = sigmoid(z);

            float error = y_pred - y[i];

            for(int j = 0; j < F; j++)
                grad_w[j] += error * X[i*F + j];

            grad_b += error;
        }

        // update pesi (una volta per epoca)
        for(int j = 0; j < F; j++)
            w[j] -= LR * (grad_w[j] / N);

        b -= LR * (grad_b / N);
    }

    auto end = chrono::high_resolution_clock::now();
    float time = chrono::duration<float, milli>(end - start).count();

    cout << "Time: " << time << " ms\n";

    int correct = 0;

    for(int i = 0; i < N; i++){

        float z = b;

        for(int j = 0; j < F; j++)
            z += X[i*F + j] * w[j];

        float y_hat = sigmoid(z);

        int prediction = (y_hat >= 0.5f) ? 1 : 0;

        if(prediction == (int)y[i])
            correct++;
    }

    float accuracy = (float)correct / N;

    printf("\nAccuracy: %.2f%%\n", accuracy * 100.0f);

    return 0;
}