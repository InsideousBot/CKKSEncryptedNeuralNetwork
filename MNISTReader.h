//
// Created by Smaran Manchala on 8/2/24.
//
#ifndef MNIST_READER_H
#define MNIST_READER_H

#include <vector>
#include <string>

using namespace std;

vector<vector<double>> readMNISTImages(const string& filename, int& numberOfImages, int& imageSize);
vector<int> readMNISTLabels(const string& filename, int& numberOfLabels);

#endif // MNIST_READER_H

