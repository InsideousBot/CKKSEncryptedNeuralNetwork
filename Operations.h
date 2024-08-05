//
// Created by Smaran Manchala on 8/2/24.
//
#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <vector>

using namespace std;

vector<vector<double>> dotProductMatrix(const vector<vector<double>>& mat1, const vector<vector<double>>& mat2);
vector<vector<double>> transposeMatrix(const vector<vector<double>>& matrix);
vector<vector<double>> generateRandomMatrix(int rows, int cols, double mean = 0.0, double stddev = 1.0);
vector<vector<double>> generateZeroMatrix(int rows, int cols);
vector<vector<double>> addBias(const vector<vector<double>>& matrix, const vector<vector<double>>& bias);
vector<vector<double>> apply_softmax_approx(const vector<vector<double>>& matrix);
vector<double> cross_entropy_loss_forward(const vector<vector<double>>& y_pred, const vector<int>& y_true);
double calculate_mean_loss(const vector<double>& sample_losses);
vector<vector<double>> sumMatrixColumns(const vector<vector<double>>& matrix);
vector<double> generate_chebyshev_coefficients_sigmoid(int degree, double a, double b);
vector<double> generate_chebyshev_coefficients_relu(int degree, double a, double b);
vector<vector<double>> chebyshev_sigmoid_approx(const vector<vector<double>>& inputs, const vector<double>& coefficients, double a, double b);
vector<vector<double>> apply_chebyshev_relu(const vector<vector<double>>& matrix, const vector<double>& coefficients, double a, double b);
#endif // MATRIX_OPERATIONS_H
