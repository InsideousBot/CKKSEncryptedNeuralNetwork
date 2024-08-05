//
// Created by Smaran Manchala on 8/2/24.
//
#include "MNISTReader.h"
#include <fstream>
#include <iostream>

using namespace std;

vector<vector<double>> readMNISTImages(const string& filename, int& numberOfImages, int& imageSize) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Could not open the file: " << filename << endl;
        exit(1);
    }

    int magicNumber = 0;
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    magicNumber = __builtin_bswap32(magicNumber);

    if (magicNumber != 2051) {
        cerr << "Invalid MNIST image file!" << endl;
        exit(1);
    }

    file.read(reinterpret_cast<char*>(&numberOfImages), sizeof(numberOfImages));
    numberOfImages = __builtin_bswap32(numberOfImages);

    int rows = 0, cols = 0;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    imageSize = rows * cols;
    vector<vector<double>> images(numberOfImages, vector<double>(imageSize));

    for (int i = 0; i < numberOfImages; ++i) {
        for (int j = 0; j < imageSize; ++j) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            images[i][j] = static_cast<double>(pixel) / 255.0;
        }
    }

    return images;
}

vector<int> readMNISTLabels(const string& filename, int& numberOfLabels) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Could not open the file: " << filename << endl;
        exit(1);
    }

    int magicNumber = 0;
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    magicNumber = __builtin_bswap32(magicNumber);

    if (magicNumber != 2049) {
        cerr << "Invalid MNIST label file!" << endl;
        exit(1);
    }

    file.read(reinterpret_cast<char*>(&numberOfLabels), sizeof(numberOfLabels));
    numberOfLabels = __builtin_bswap32(numberOfLabels);

    vector<int> labels(numberOfLabels);

    for (int i = 0; i < numberOfLabels; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = static_cast<int>(label);
    }

    return labels;
}

