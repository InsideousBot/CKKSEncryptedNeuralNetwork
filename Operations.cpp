//
// Created by Smaran Manchala on 8/2/24.
//
#include "Operations.h"
#include <random>

using namespace std;

vector<vector<double>> dotProductMatrix(const vector<vector<double>>& mat1, const vector<vector<double>>& mat2) {
    int rows1 = mat1.size();
    int cols1 = mat1[0].size();
    int cols2 = mat2[0].size();

    vector<vector<double>> result(rows1, vector<double>(cols2, 0));

    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            for (int k = 0; k < cols1; ++k) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }

    return result;
}

vector<vector<double>> transposeMatrix(const vector<vector<double>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    vector<vector<double>> transposed(cols, vector<double>(rows, 0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}

vector<vector<double>> generateRandomMatrix(int rows, int cols, double mean, double stddev) {
    vector<vector<double>> matrix(rows, vector<double>(cols));

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> d(mean, stddev);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = d(gen);
        }
    }

    return matrix;
}

vector<vector<double>> generateZeroMatrix(int rows, int cols) {
    return vector<vector<double>>(rows, vector<double>(cols, 0.0));
}

vector<vector<double>> addBias(const vector<vector<double>>& matrix, const vector<vector<double>>& bias) {
    int batchSize = matrix.size();
    int neurons = matrix[0].size();

    if (bias.size() != 1 || bias[0].size() != neurons) {
        throw invalid_argument("Bias dimensions must be 1 x neurons.");
    }

    vector<vector<double>> result = matrix;

    for (int i = 0; i < batchSize; ++i) {
        for (int j = 0; j < neurons; ++j) {
            result[i][j] += bias[0][j];
        }
    }

    return result;
}

vector<double> softmax_approx(const vector<double>& logits) {
    vector<double> poly_approx(logits.size());

    // Polynomial approximation for each logit
    transform(logits.begin(), logits.end(), poly_approx.begin(), [](double logit) {
        return 1.0 + logit + 0.5 * logit * logit;
    });

    double sum_approx = accumulate(poly_approx.begin(), poly_approx.end(), 0.0);

    // Normalizing to get probabilities
    vector<double> probabilities(logits.size());
    transform(poly_approx.begin(), poly_approx.end(), probabilities.begin(), [sum_approx](double approx) {
        return approx / sum_approx;
    });

    return probabilities;
}

// Function to apply softmax approximation to each row of a 2D matrix
vector<vector<double>> apply_softmax_approx(const vector<vector<double>>& matrix) {
    vector<vector<double>> result(matrix.size(), vector<double>(matrix[0].size()));
    for (size_t i = 0; i < matrix.size(); ++i) {
        result[i] = softmax_approx(matrix[i]);
    }
    return result;
}


// Taylor series approximation of the natural logarithm function
double taylor_log(double x) {
    double y = x - 1;
    return y - (y * y / 2) + (y * y * y / 3) - (y * y * y * y / 4);
}

// Function to calculate the cross-entropy loss for each sample
vector<double> cross_entropy_loss_forward(const vector<vector<double>>& y_pred, const vector<int>& y_true) {
    size_t samples = y_pred.size();
    vector<double> losses(samples);

    for (size_t i = 0; i < samples; ++i) {
        double correct_confidence = y_pred[i][y_true[i]];
        losses[i] = -taylor_log(correct_confidence);
    }

    return losses;
}

// Function to calculate the mean loss over a batch
double calculate_mean_loss(const vector<double>& sample_losses) {
    double sum_loss = 0.0;
    for (double loss : sample_losses) {
        sum_loss += loss;
    }
    return sum_loss / sample_losses.size();
}

vector<vector<double>> sumMatrixColumns(const vector<vector<double>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<vector<double>> columnSums(1, vector<double>(cols, 0.0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            columnSums[0][j] += matrix[i][j];
        }
    }

    return columnSums;
}

vector<double> generate_chebyshev_coefficients_sigmoid(int degree, double a, double b) {
    vector<double> coefficients(degree + 1);
    for (int k = 0; k <= degree; ++k) {
        double sum = 0.0;
        for (int n = 0; n <= degree; ++n) {
            double x = cos(M_PI * (n + 0.5) / (degree + 1));
            double fx = 1 / (1 + exp(-x * (b - a) / 2 - (b + a) / 2));
            sum += fx * cos(M_PI * k * (n + 0.5) / (degree + 1));
        }
        coefficients[k] = (2.0 / (degree + 1)) * sum;
    }
    return coefficients;
}

vector<double> generate_chebyshev_coefficients_sigmoid_derivative(int degree, double a, double b) {
    vector<double> coefficients(degree + 1);
    for (int k = 0; k <= degree; ++k) {
        double sum = 0.0;
        for (int n = 0; n <= degree; ++n) {
            double x = cos(M_PI * (n + 0.5) / (degree + 1));
            double fx = 1 / (1 + exp(-x * (b - a) / 2 - (b + a) / 2));
            double fx_derivative = fx * (1 - fx);
            sum += fx_derivative * cos(M_PI * k * (n + 0.5) / (degree + 1));
        }
        coefficients[k] = (2.0 / (degree + 1)) * sum;
    }
    return coefficients;
}

vector<vector<double>> chebyshev_sigmoid_approx(const vector<vector<double>>& inputs, const vector<double>& coefficients, double a, double b) {
    int degree = coefficients.size() - 1;
    vector<vector<double>> outputs(inputs.size(), vector<double>(inputs[0].size(), 0.0));

    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t j = 0; j < inputs[0].size(); ++j) {
            double x = inputs[i][j];
            double t = (2 * x - a - b) / (b - a); // Map x to [-1, 1]
            double y = coefficients[degree];
            for (int k = degree - 1; k >= 0; --k) {
                y = y * t + coefficients[k];
            }
            outputs[i][j] = y;
        }
    }

    return outputs;
}

vector<vector<double>> chebyshev_sigmoid_derivative_approx(const vector<vector<double>>& inputs, const vector<double>& coefficients, double a, double b) {
    int degree = coefficients.size() - 1;
    vector<vector<double>> outputs(inputs.size(), vector<double>(inputs[0].size(), 0.0));

    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t j = 0; j < inputs[0].size(); ++j) {
            double x = inputs[i][j];
            double t = (2 * x - a - b) / (b - a); // Map x to [-1, 1]
            double y = coefficients[degree];
            for (int k = degree - 1; k >= 0; --k) {
                y = y * t + coefficients[k];
            }
            outputs[i][j] = y;
        }
    }

    return outputs;
}

vector<double> generate_chebyshev_coefficients(int degree, double a, double b) {
    // Helper function to generate Chebyshev nodes
    auto chebyshev_nodes = [](int n) {
        vector<double> nodes(n);
        for (int k = 0; k < n; ++k) {
            nodes[k] = cos((2.0 * k + 1) / (2.0 * n) * M_PI);
        }
        return nodes;
    };

    // Helper function to compute Chebyshev coefficients for the ReLU function
    auto compute_coefficients = [&](int degree, const vector<double>& nodes, double a, double b) {
        int n = nodes.size();
        vector<double> f_values(n);
        for (int i = 0; i < n; ++i) {
            double x_mapped = 0.5 * (nodes[i] * (b - a) + (a + b));
            f_values[i] = max(0.0, x_mapped);
        }

        vector<double> coefficients(degree + 1, 0.0);
        for (int k = 0; k <= degree; ++k) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                sum += f_values[j] * cos(k * acos(nodes[j]));
            }
            coefficients[k] = (2.0 / n) * sum;
        }
        coefficients[0] /= 2.0;
        return coefficients;
    };

    // Generate Chebyshev nodes
    vector<double> nodes = chebyshev_nodes(degree + 1);

    // Compute and return Chebyshev coefficients
    return compute_coefficients(degree, nodes, a, b);
}

double chebyshev_relu(double x, const vector<double>& coefficients, double a, double b) {
    int degree = coefficients.size() - 1;

    // Map input x from [a, b] to [-1, 1]
    double x_mapped = (2.0 * (x - a) / (b - a)) - 1.0;

    // Evaluate Chebyshev polynomial at x
    double sum = coefficients[0];
    double T_prev = 1.0;
    double T_curr = x_mapped;

    for (int k = 1; k <= degree; ++k) {
        double T_next = 2 * x_mapped * T_curr - T_prev;
        sum += coefficients[k] * T_curr;
        T_prev = T_curr;
        T_curr = T_next;
    }

    return sum;
}

vector<vector<double>> apply_chebyshev_relu(const vector<vector<double>>& matrix, const vector<double>& coefficients, double a, double b) {
    vector<vector<double>> result(matrix.size(), vector<double>(matrix[0].size()));

    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            result[i][j] = chebyshev_relu(matrix[i][j], coefficients, a, b);
        }
    }

    return result;
}