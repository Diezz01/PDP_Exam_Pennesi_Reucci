#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

struct Car {
    int year;
    std::string make;
    std::string model;
};

std::vector<Car> loadCSV(const std::string& filename) {

    std::vector<Car> dataset;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Errore apertura file\n";
        return dataset;
    }

    std::getline(file, line); // salta header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        Car car;

        std::getline(ss, value, ',');
        car.year = std::stoi(value);

        std::getline(ss, car.make, ',');
        std::getline(ss, car.model, ',');

        dataset.push_back(car);
    }

    return dataset;
}

int main() {
    std::string filename = "dataset.csv.xlsx";
    std::ifstream file(filename);
    std::vector<Car> dataset = loadCSV(filename);
   
    std::cout << "Numero righe: " << dataset.size() << std::endl;

    return 0;
}