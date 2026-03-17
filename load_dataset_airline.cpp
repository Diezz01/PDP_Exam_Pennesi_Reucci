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

// Function that decides if a column should be normalized or not
bool shouldNormalize(uint32_t j){
    if (j == 2 || j == 7 || j == 22 || j == 23)
        return true;
    return false;
}

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

    uint32_t n_features = 22;
    uint32_t n_samples = 0;

    while (getline(file, line)) {

        stringstream ss(line);
        string token;

        // Target
        getline(ss, token, ';');

        if (token == "satisfied")
            y.push_back(1.0f);
        else
            y.push_back(0.0f);

        // Gender
        getline(ss, token, ';');
        if (token == "Male")
            X.push_back(1.0f);
        else
            X.push_back(0.0f);

        // Customer Type
        getline(ss, token, ';');
        if (token == "Loyal Customer")
            X.push_back(1.0f);
        else
            X.push_back(0.0f);

        // Age
        getline(ss, token, ';');
        X.push_back(stof(token));

        // Type of travel
        getline(ss, token, ';');
        if (token == "Business travel")
            X.push_back(1.0f);
        else
            X.push_back(0.0f);

        // Class (one hot)
        getline(ss, token, ';');

        if (token == "Business") {
            X.push_back(1.0f);
            X.push_back(0.0f);
            X.push_back(0.0f);
        }
        else if (token == "Eco") {
            X.push_back(0.0f);
            X.push_back(1.0f);
            X.push_back(0.0f);
        }
        else if (token == "Eco Plus") {
            X.push_back(0.0f);
            X.push_back(0.0f);
            X.push_back(1.0f);
        }
        else {
            X.push_back(0.0f);
            X.push_back(0.0f);
            X.push_back(0.0f);
        }

        // Remaining features
        for (uint32_t i = 0; i < 17; i++) {

            getline(ss, token, ';');

            if (token.empty())
                token = "0";

            X.push_back(stof(token));
        }

        n_samples++;
    }

    n_features += 2; // one-hot encode

    file.close();
    cout << "X size: " << X.size() << endl;
    cout << "Expected: " << n_samples * n_features << endl;

    cout << "Campioni: " << n_samples << endl;
    cout << "Feature: " << n_features << endl;

    // ------------------------
    // Computing mean
    // ------------------------

    vector<float> mean(n_features, 0.0f);
    vector<float> stddev(n_features, 0.0f);

    for (uint32_t i = 0; i < n_samples; i++) {
        for (uint32_t j = 0; j < n_features; j++) {

            mean[j] += X[i * n_features + j];

        }
    }

    for (uint32_t j = 0; j < n_features; j++) {

        mean[j] /= n_samples;

    }

    // ------------------------
    // Computing std
    // ------------------------

    for (uint32_t i = 0; i < n_samples; i++) {
        for (uint32_t j = 0; j < n_features; j++) {

            float diff = X[i * n_features + j] - mean[j];
            stddev[j] += diff * diff;

        }
    }

    for (uint32_t j = 0; j < n_features; j++) {

        stddev[j] = sqrt(stddev[j] / n_samples);

        if (stddev[j] == 0.0f)
            stddev[j] = 1.0f;
    }

    // ------------------------
    // Normalization
    // ------------------------

    for (uint32_t i = 0; i < n_samples; i++) {
        for (uint32_t j = 0; j < n_features; j++) {

            if (shouldNormalize(j)) {

                X[i * n_features + j] =
                    (X[i * n_features + j] - mean[j]) / stddev[j];

            }
        }
    }

    // ------------------------
    // Writing of normalized CSV
    // ------------------------

    ofstream csv_out("dataset_normalized.csv");

    if (!csv_out.is_open()) {
        cerr << "Errore apertura CSV output\n";
        return 1;
    }

    csv_out << "target";

    for (uint32_t j = 0; j < n_features; j++) {

        csv_out << ";f" << j;

    }

    csv_out << "\n";

    for (uint32_t i = 0; i < n_samples; i++) {

        csv_out << y[i];

        for (uint32_t j = 0; j < n_features; j++) {

            csv_out << ";" << X[i * n_features + j];

        }

        csv_out << "\n";
    }

    csv_out.close();

    cout << "File dataset_normalized.csv scritto correttamente\n";

    // ------------------------
    // Writing BINARY file
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

    cout << "File dataset_normalized.bin scritto correttamente\n";

    return 0;
}