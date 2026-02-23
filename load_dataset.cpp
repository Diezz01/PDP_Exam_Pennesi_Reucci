#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
using namespace std;

constexpr int NUM_FEATURES = 10;

struct Header {
    uint32_t n_samples;
    uint32_t n_features;
};

int encode(unordered_map<string, int>& map, const string& key) {
    auto it = map.find(key);
    if (it != map.end()) return it->second;
    int id = map.size();
    map[key] = id;
    return id;
}

static inline string clean(string s) {
    // rimuove spazi iniziali e finali
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
        getline(ss, token, ';');
        token = clean(token);
        X.push_back(stof(token));
        
        // make
        getline(ss, token, ';');
        token = clean(token);
        X.push_back(encode(make_map, token));
       
        // model
        getline(ss, token, ';');
        token = clean(token);
        X.push_back(encode(model_map, token));

        // body
        getline(ss, token, ';');
        token = clean(token);
        X.push_back(encode(body_map, token));

        // transmission
        getline(ss, token, ';');
        token = clean(token);
        if (token.empty()) token = "-1"; // missing value
        X.push_back(encode(trans_map, token));

        // state
        getline(ss, token, ';');
        token = clean(token);
        X.push_back(encode(state_map, token));

        // condition
        getline(ss, token, ';');
        token = clean(token);
        if (token.empty()) token = "-1"; // missing value
        X.push_back(stof(token));

        // odometer
        getline(ss, token, ';');
        token = clean(token);
        if (token.empty()) token = "-1"; // missing value
        X.push_back(stof(token));

        // color
        getline(ss, token, ';');
        token = clean(token);
        if (token.empty()) token = "-1"; // missing value
        X.push_back(encode(color_map, token));

        // mmr
        getline(ss, token, ';');
        token = clean(token);
        if (token.empty()) token = "-1";
        X.push_back(stof(token));

        // sellingprice → y
        getline(ss, token, '\n');
        token = clean(token);
        if (token.empty()) token = "-1";
        y.push_back(stof(token));

        ++N;
    }

    ofstream out("dataset.bin", ios::binary);

    Header h{(uint32_t)N, NUM_FEATURES};
    out.write((char*)&h, sizeof(h));
    out.write((char*)X.data(), X.size() * sizeof(float));
    out.write((char*)y.data(), y.size() * sizeof(float));

    cout << "Righe: " << N << "\n";
    cout << "Categorie uniche:\n";
    cout << " make: " << make_map.size() << "\n";
    cout << " model: " << model_map.size() << "\n";
    cout << " body: " << body_map.size() << "\n";
    cout << " transmission: " << trans_map.size() << "\n";
    cout << " state: " << state_map.size() << "\n";
    cout << " color: " << color_map.size() << "\n";

    auto save_map = [](const unordered_map<string,int>& map, const string& filename){
        ofstream out(filename);
        for (auto &p : map) out << p.first << " " << p.second << "\n";
    };

    save_map(make_map, "make_map.txt");
    save_map(model_map, "model_map.txt");
    save_map(body_map, "body_map.txt");
    save_map(trans_map, "trans_map.txt");
    save_map(state_map, "state_map.txt");
    save_map(color_map, "color_map.txt");

    return 0;
}