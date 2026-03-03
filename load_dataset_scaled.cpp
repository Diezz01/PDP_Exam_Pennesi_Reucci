#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <cmath>
using namespace std;

constexpr int NUM_FEATURES = 10;

struct Header {
    uint32_t n_samples;
    uint32_t n_features;
};

// Codifica le feature categoriche
int encode(unordered_map<string, int>& map, const string& key) {
    auto it = map.find(key);
    if (it != map.end()) return it->second;
    int id = map.size();
    map[key] = id;
    return id;
}

// Rimuove spazi, doppi apici e \r
static inline string clean(string s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '"')) s.erase(s.begin());
    while (!s.empty() && (s.back()  == ' ' || s.back()  == '"' || s.back() == '\r')) s.pop_back();
    return s;
}

int main() {
    cout << "Loading dataset...\n";
    ifstream file("dataset.csv");
    if (!file) {
        cerr << "Errore apertura dataset.csv\n";
        return 1;
    }

    string line;
    getline(file, line); // salta header

    unordered_map<string,int> make_map, model_map, body_map,
                             trans_map, state_map, color_map;

    vector<float> X;
    vector<float> y;

    X.reserve(558837 * NUM_FEATURES);
    y.reserve(558837);

    size_t N = 0;
    while (getline(file, line)) {
        stringstream ss(line);
        string token;

        // year
        getline(ss, token, ';'); token = clean(token);
        X.push_back(stof(token));
        
        // make
        getline(ss, token, ';'); token = clean(token);
        X.push_back((float)encode(make_map, token));
       
        // model
        getline(ss, token, ';'); token = clean(token);
        X.push_back((float)encode(model_map, token));

        // body
        getline(ss, token, ';'); token = clean(token);
        X.push_back((float)encode(body_map, token));

        // transmission
        getline(ss, token, ';'); token = clean(token);
        if (token.empty()) token = "-1";
        X.push_back((float)encode(trans_map, token));

        // state
        getline(ss, token, ';'); token = clean(token);
        X.push_back((float)encode(state_map, token));

        // condition
        getline(ss, token, ';'); token = clean(token);
        if (token.empty()) token = "-1";
        X.push_back(stof(token));

        // odometer
        getline(ss, token, ';'); token = clean(token);
        if (token.empty()) token = "-1";
        X.push_back(stof(token));

        // color
        getline(ss, token, ';'); token = clean(token);
        if (token.empty()) token = "-1";
        X.push_back((float)encode(color_map, token));

        // mmr
        getline(ss, token, ';'); token = clean(token);
        if (token.empty()) token = "-1";
        X.push_back(stof(token));

        // sellingprice → y
        getline(ss, token, '\n'); token = clean(token);
        if (token.empty()) token = "-1";
        y.push_back(stof(token));

        ++N;
    }

    file.close();
    cout << "Righe lette: " << N << "\n";

    // --- Normalizzazione numerica ---
    int idx_num[] = {0,6,7,9}; // year, condition, odometer, mmr
    vector<float> mean(NUM_FEATURES, 0.0f);
    vector<float> stddev(NUM_FEATURES, 0.0f);

    // media
    for (int j : idx_num)
        for (size_t i = 0; i < N; i++)
            mean[j] += X[i*NUM_FEATURES + j];
    for (int j : idx_num)
        mean[j] /= N;

    // deviazione standard
    for (int j : idx_num)
        for (size_t i = 0; i < N; i++) {
            float diff = X[i*NUM_FEATURES + j] - mean[j];
            stddev[j] += diff * diff;
        }
    for (int j : idx_num)
        stddev[j] = sqrt(stddev[j] / N);

    // normalizzazione Z-score
    for (int j : idx_num)
        for (size_t i = 0; i < N; i++)
            X[i*NUM_FEATURES + j] = (X[i*NUM_FEATURES + j] - mean[j]) / (stddev[j] + 1e-8f);

    // --- Scaling feature categoriche [0,1] ---
    vector<int> max_ids = {
        (int)make_map.size()-1,
        (int)model_map.size()-1,
        (int)body_map.size()-1,
        (int)trans_map.size()-1,
        (int)state_map.size()-1,
        (int)color_map.size()-1
    };

    int idx_cat[] = {1,2,3,4,5,8}; // make, model, body, transmission, state, color
    for (size_t i = 0; i < N; i++)
        for (size_t k = 0; k < 6; k++)
            X[i*NUM_FEATURES + idx_cat[k]] /= max_ids[k];

    // --- Salvataggio in binario ---
    ofstream out("dataset.bin", ios::binary);
    Header h{(uint32_t)N, NUM_FEATURES};
    out.write((char*)&h, sizeof(h));
    out.write((char*)X.data(), X.size() * sizeof(float));
    out.write((char*)y.data(), y.size() * sizeof(float));
    out.close();

    cout << "Dataset normalizzato e salvato in dataset.bin\n";
    return 0;
}