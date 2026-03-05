#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
using namespace std;

struct Header {
    uint32_t n_samples;
    uint32_t n_features;
};

int main() {
    ifstream file("Dataset_airline.csv");
    if (!file.is_open()) {
        cerr << "Errore apertura CSV\n";
        return 1;
    }

    string line;
    getline(file, line); // skip header

    vector<float> X;
    vector<float> y;

    uint32_t n_features = 22; // senza timestamp
    uint32_t n_samples = 0;

    while (getline(file, line)) {
        stringstream ss(line);
        string token;

        // 1) reading target
        getline(ss, token, ';');
        if(token == "satisfied") y.push_back(1.0f);
        else if(token == "dissatisfied") y.push_back(0.0f);
        
        // 2) Gender
        getline(ss, token, ';');
        if(token == "Male") X.push_back(1.0f);
        else if(token == "Female") X.push_back(0.0f);

        // 3) Customer Type
        getline(ss, token, ';');
        if(token == "Loyal Customer") X.push_back(1.0f);
        else if(token == "disloyal Customer") X.push_back(0.0f);

        // 4) Age
        getline(ss, token, ';');
        X.push_back(stof(token));
        
        // 5) Type of Travel
        getline(ss, token, ';');
        if(token == "Business travel") X.push_back(1.0f);
        else if(token == "Personal Travel") X.push_back(0.0f);

        // 6) Class TODO: one hot encoding
        getline(ss, token, ';');
        if(token == "Business"){
            X.push_back(1.0f); // business
            X.push_back(0.0f); // eco
            X.push_back(0.0f); // eco plus
        } else if(token == "Eco"){
            X.push_back(0.0f);
            X.push_back(1.0f);
            X.push_back(0.0f);
        } else if(token == "Eco Plus"){
            X.push_back(0.0f);
            X.push_back(0.0f);
            X.push_back(1.0f);
        } else {
            // default, oppure errore
            X.push_back(0.0f);
            X.push_back(0.0f);
            X.push_back(0.0f);
        }
        
        // 7) Other 17 features
        for (uint32_t i = 0; i < 17; i++) {
            getline(ss, token, ';');
            if(token.empty()) token = "0"; // handle missing values
            X.push_back(stof(token));
        }

        n_samples++;
    }

    n_features += 2; //adding columns for one hot encode

    file.close();

    cout << "Campioni: " << n_samples << "\n";
    cout << "Feature: " << n_features << "\n";

    // ------------------------
    // Calcolo mean
    // ------------------------
    vector<float> mean(n_features, 0.0f);
    vector<float> stddev(n_features, 0.0f);

    for (uint32_t i = 0; i < n_samples; ++i) {
        for (uint32_t j = 0; j < n_features; ++j) {
            mean[j] += X[i * n_features + j];
        }
    }

    for (uint32_t j = 0; j < n_features; ++j) {
        mean[j] /= n_samples;
    }

    // ------------------------
    // Calcolo std
    // ------------------------
    for (uint32_t i = 0; i < n_samples; ++i) {
        for (uint32_t j = 0; j < n_features; ++j) {
            float diff = X[i * n_features + j] - mean[j];
            stddev[j] += diff * diff;
        }
    }

    for (uint32_t j = 0; j < n_features; ++j) {
        stddev[j] = sqrt(stddev[j] / n_samples);
        if (stddev[j] == 0.0f)
            stddev[j] = 1.0f;
    }

    // ------------------------
    // Normalizzazione
    // ------------------------
    for (uint32_t i = 0; i < n_samples; ++i) {
        for (uint32_t j = 0; j < n_features; ++j) {
            X[i * n_features + j] =
                (X[i * n_features + j] - mean[j]) / stddev[j];
        }
    }

    // ------------------------
    // Scrittura file binario
    // ------------------------
    ofstream out("dataset_normalized.bin", ios::binary);

    Header header;
    header.n_samples = n_samples;
    header.n_features = n_features;

    out.write(reinterpret_cast<char*>(&header), sizeof(Header));
    out.write(reinterpret_cast<char*>(mean.data()), n_features * sizeof(float));
    out.write(reinterpret_cast<char*>(stddev.data()), n_features * sizeof(float));
    out.write(reinterpret_cast<char*>(X.data()), X.size() * sizeof(float));
    out.write(reinterpret_cast<char*>(y.data()), y.size() * sizeof(float));

    out.close();

    cout << "File dataset_normalized.bin scritto correttamente.\n";

    return 0;
}