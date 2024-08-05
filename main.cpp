//
// Created by Smaran Manchala on 8/2/24.
//
#include <iostream>
#include <fstream>
#include "Operations.h"
#include "MNISTReader.h"
#include <chrono>
#include <filesystem>
#include <thread>
#include <mutex>

using namespace std;
using namespace chrono;
namespace fs = filesystem;

double calculate_accuracy(const vector<vector<double>>& outputs, const vector<int>& labels) {
    int correct_predictions = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto max_elem = max_element(outputs[i].begin(), outputs[i].end());
        int predicted_label = distance(outputs[i].begin(), max_elem);
        if (predicted_label == labels[i]) {
            ++correct_predictions;
        }
    }
    return static_cast<double>(correct_predictions) / outputs.size();
}

vector<vector<double>> forward_hidden_layer(const vector<vector<double>>& inputs,
                                            const vector<vector<double>>& weights, const vector<vector<double>>& biases) {
    vector<vector<double>> outputs = dotProductMatrix(inputs, weights);
    outputs = addBias(outputs, biases);
    return outputs;
}

vector<vector<double>> forward_sigmoid(const vector<vector<double>>& inputs,
                                          const vector<double>& chebyshev_coefficients, double chebyshev_a, double chebyshev_b) {
    vector<vector<double>> outputs = chebyshev_sigmoid_approx(inputs, chebyshev_coefficients, chebyshev_a, chebyshev_b);
    return outputs;
}

vector<vector<double>> forward_relu(const vector<vector<double>>& inputs,
                                    const vector<double>& chebyshev_coefficients, double chebyshev_a, double chebyshev_b) {
    return apply_chebyshev_relu(inputs, chebyshev_coefficients, chebyshev_a, chebyshev_b);
}

vector<vector<double>> forward_output_layer(const vector<vector<double>>& inputs,
                                            const vector<vector<double>>& weights, const vector<vector<double>>& biases) {
    vector<vector<double>> outputs = dotProductMatrix(inputs, weights);
    outputs = addBias(outputs, biases);
    return outputs;
}

vector<vector<double>> forward_softmax(const vector<vector<double>>& logits) {
    vector<vector<double>> outputs = apply_softmax_approx(logits);
    return outputs;
}

vector<vector<double>> backward_softmax_cross_entropy(const vector<vector<double>>& dvalues, const vector<int>& y_true) {
    int samples = dvalues.size();
    int labels = dvalues[0].size();

    // Initialize the gradient matrix
    vector<vector<double>> dinputs = dvalues;

    // Convert one-hot encoded labels to discrete values if necessary
    vector<int> y_true_discrete = y_true;
    if (y_true[0] < 0 || y_true[0] > labels) {
        for (int i = 0; i < samples; ++i) {
            y_true_discrete[i] = distance(y_true.begin(), max_element(y_true.begin(), y_true.end()));
        }
    }

    // Calculate gradient
    for (int i = 0; i < samples; ++i) {
        dinputs[i][y_true_discrete[i]] -= 1.0;
    }

    // Normalize gradient
    for (int i = 0; i < samples; ++i) {
        for (int j = 0; j < labels; ++j) {
            dinputs[i][j] /= samples;
        }
    }
    return dinputs;
}

vector<vector<double>> backward_layer(const vector<vector<double>>& dvalues,
                                      const vector<vector<double>>& inputs,
                                      const vector<vector<double>>& weights,
                                      vector<vector<double>>& dweights,
                                      vector<vector<double>>& dbiases) {
    // Calculate gradients on parameters
    dweights = dotProductMatrix(transposeMatrix(inputs), dvalues);
    dbiases = sumMatrixColumns(dvalues);  // Sum along the samples axis

    // Calculate gradient on values
    vector<vector<double>> dinputs = dotProductMatrix(dvalues, transposeMatrix(weights));

    return dinputs;
}

vector<vector<double>> backward_sigmoid(const vector<vector<double>>& dvalues, const vector<vector<double>>& inputs,
                                        const vector<double>& chebyshev_coefficients, double chebyshev_a, double chebyshev_b) {
    vector<vector<double>> sigmoid_approx = chebyshev_sigmoid_approx(inputs, chebyshev_coefficients, chebyshev_a, chebyshev_b);
    vector<vector<double>> dinputs(dvalues.size(), vector<double>(dvalues[0].size(), 0.0));

    for (size_t i = 0; i < dinputs.size(); ++i) {
        for (size_t j = 0; j < dinputs[0].size(); ++j) {
            dinputs[i][j] = dvalues[i][j] * sigmoid_approx[i][j];
        }
    }
    return dinputs;
}

void update_params(vector<vector<double>>& weights, vector<vector<double>>& dweights,
                   vector<vector<double>>& biases, vector<vector<double>>& dbiases, double learning_rate) {
    // Update weights and biases using SGD
    if (weights.size() != dweights.size() || weights[0].size() != dweights[0].size()) {
        cerr << "Dimension mismatch between weights and dweights." << endl;
        exit(1);
    }
    if (biases.size() != dbiases.size() || biases[0].size() != dbiases[0].size()) {
        cerr << "Dimension mismatch between biases and dbiases." << endl;
        exit(1);
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[0].size(); ++j) {
            weights[i][j] -= learning_rate * dweights[i][j];
        }
    }
    for (size_t i = 0; i < biases.size(); ++i) {
        for (size_t j = 0; j < biases[0].size(); ++j) {
            biases[i][j] -= learning_rate * dbiases[i][j];
        }
    }
}

pair<double, double> test(const vector<vector<double>>& weights1, const vector<vector<double>>& bias1,
                          const vector<vector<double>>& weights2, const vector<vector<double>>& bias2,
                          const vector<double>& chebyshev_relu_coefficients, double chebyshev_a, double chebyshev_b,
                          const vector<vector<double>>& testImages, const vector<int>& testLabels) {
    // Forward pass through the hidden layer
    vector<vector<double>> hidden_layer_outputs = forward_hidden_layer(testImages, weights1, bias1);

    // Apply the sigmoid activation function
    vector<vector<double>> relu_outputs = forward_relu(hidden_layer_outputs, chebyshev_relu_coefficients, chebyshev_a, chebyshev_b);

    // Forward pass through the output layer
    vector<vector<double>> output_layer_outputs = forward_output_layer(relu_outputs, weights2, bias2);

    // Apply the softmax function
    vector<vector<double>> softmax_outputs = forward_softmax(output_layer_outputs);

    // Calculate the loss
    vector<double> sample_losses = cross_entropy_loss_forward(softmax_outputs, testLabels);
    double test_loss = calculate_mean_loss(sample_losses);

    // Calculate accuracy
    double test_accuracy = calculate_accuracy(softmax_outputs, testLabels);

    return make_pair(test_loss, test_accuracy);
}

struct NetworkParameters {
    vector<vector<double>> weights1;
    vector<vector<double>> bias1;
    vector<vector<double>> weights2;
    vector<vector<double>> bias2;
};

mutex file_mutex;
mutex console_mutex;

NetworkParameters train(int batch_size, int epochs,
                        vector<vector<double>>& trainImages, vector<int>& trainLabels,
                        vector<vector<double>>& testImages, vector<int>& testLabels,
                        int input_length, int neurons, int num_outputs,
                        const vector<double>& chebyshev_relu_coefficients, const vector<double>& chebyshev_sigmoid_coefficients,
                        double chebyshev_a, double chebyshev_b,
                        double learning_rate,
                        double& final_test_loss, double& final_test_accuracy) {
    int numberOfImages = trainImages.size();
    cout << "Loop starting" << endl;

    // Initializing Weights and Biases
    vector<vector<double>> weights1 = generateRandomMatrix(input_length, neurons, 0.0, 1.0);
    vector<vector<double>> bias1 = generateZeroMatrix(1, neurons);
    vector<vector<double>> weights2 = generateRandomMatrix(neurons, num_outputs, 0.0, 1.0);
    vector<vector<double>> bias2 = generateZeroMatrix(1, num_outputs);

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;
        double epoch_accuracy = 0.0;
        for (int i = 0; i < numberOfImages; i += batch_size) {
            // Prepare batch
            vector<vector<double>> inputs(trainImages.begin() + i, trainImages.begin() + min(i + batch_size, numberOfImages));
            vector<int> batch_labels(trainLabels.begin() + i, trainLabels.begin() + min(i + batch_size, numberOfImages));

            // Forward pass
            vector<vector<double>> hidden_layer_outputs = forward_hidden_layer(inputs, weights1, bias1);
            vector<vector<double>> relu_outputs = forward_relu(hidden_layer_outputs, chebyshev_relu_coefficients, chebyshev_a, chebyshev_b);
            vector<vector<double>> output_layer_outputs = forward_output_layer(relu_outputs, weights2, bias2);
            vector<vector<double>> softmax_outputs = forward_softmax(output_layer_outputs);

            // Calculate loss
            vector<double> sample_losses = cross_entropy_loss_forward(softmax_outputs, batch_labels);
            epoch_loss += calculate_mean_loss(sample_losses);

            // Calculate accuracy
            double batch_accuracy = calculate_accuracy(softmax_outputs, batch_labels);
            epoch_accuracy += batch_accuracy;

            // Backward pass
            vector<vector<double>> dSoftmaxCrossEntropyInputs = backward_softmax_cross_entropy(softmax_outputs, batch_labels);
            vector<vector<double>> dweights2(weights2.size(), vector<double>(weights2[0].size(), 0.0));
            vector<vector<double>> dbiases2(1, vector<double>(bias2[0].size(), 0.0));
            vector<vector<double>> dinputs2 = backward_layer(dSoftmaxCrossEntropyInputs, relu_outputs, weights2, dweights2, dbiases2);

            vector<vector<double>> dSigmoidInputs = backward_sigmoid(dinputs2, hidden_layer_outputs, chebyshev_sigmoid_coefficients, chebyshev_a, chebyshev_b);

            vector<vector<double>> dweights1(weights1.size(), vector<double>(weights1[0].size(), 0.0));
            vector<vector<double>> dbiases1(1, vector<double>(bias1[0].size(), 0.0));
            vector<vector<double>> dinputs1 = backward_layer(dSigmoidInputs, inputs, weights1, dweights1, dbiases1);

            update_params(weights2, dweights2, bias2, dbiases2, learning_rate);
            update_params(weights1, dweights1, bias1, dbiases1, learning_rate);
        }
        epoch_loss /= (numberOfImages / batch_size);
        epoch_accuracy /= (numberOfImages / batch_size);

        // Test the model
        tie(final_test_loss, final_test_accuracy) = test(weights1, bias1, weights2, bias2, chebyshev_relu_coefficients, chebyshev_a, chebyshev_b, testImages, testLabels);
    }
    return {weights1, bias1, weights2, bias2};
}

void train_and_save(int n, int batch_size, int epochs,
                    vector<vector<double>>& trainImages, vector<int>& trainLabels,
                    vector<vector<double>>& testImages, vector<int>& testLabels,
                    int input_length, int neurons, int num_outputs,
                    const vector<double>& chebyshev_relu_coefficients, const vector<double>& chebyshev_sigmoid_coefficients,
                    double chebyshev_a, double chebyshev_b,
                    double learning_rate, int num_networks) {
    string filename = "training_results_" + to_string(n + 1) + ".txt";
    ofstream results_file(filename);
    if (!results_file.is_open()) {
        cerr << "Unable to open file for writing: " << filename << endl;
        return;
    }

    {
        lock_guard<mutex> guard(console_mutex);
        cout << "Training network " << n + 1 << " of " << num_networks << endl;
    }

    {
        lock_guard<mutex> guard(console_mutex);
        cout << "Loop starting for network " << n + 1 << endl;
    }

    auto start_time = high_resolution_clock::now();

    // Train the network
    double final_loss;
    double final_accuracy;
    NetworkParameters params = train(batch_size, epochs, trainImages, trainLabels, testImages, testLabels,
                                     input_length, neurons, num_outputs, chebyshev_relu_coefficients, chebyshev_sigmoid_coefficients,
                                     chebyshev_a, chebyshev_b, learning_rate, final_loss, final_accuracy);

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(end_time - start_time).count();

    // Write the results to the file
    results_file << "Training network " << n + 1 << " of " << num_networks << "\n";
    results_file << "Training Time: " << duration << " seconds\n";
    results_file << "Epochs: " << epochs << "\n";
    results_file << "Batch Size: " << batch_size << "\n";
    results_file << "Learning Rate: " << learning_rate << "\n";
    results_file << "Final Test Loss: " << final_loss << "\n";
    results_file << "Final Test Accuracy: " << final_accuracy << "\n";

    // Save final weights and biases (Do later, no print to print all)
    /*results_file << "Weights1:\n";
    for (const auto& row : params.weights1) {
        for (const auto& val : row) {
            results_file << val << " ";
        }
        results_file << "\n";
    }
    results_file << "Bias1:\n";
    for (const auto& row : params.bias1) {
        for (const auto& val : row) {
            results_file << val << " ";
        }
        results_file << "\n";
    }
    results_file << "Weights2:\n";
    for (const auto& row : params.weights2) {
        for (const auto& val : row) {
            results_file << val << " ";
        }
        results_file << "\n";
    }
    results_file << "Bias2:\n";
    for (const auto& row : params.bias2) {
        for (const auto& val : row) {
            results_file << val << " ";
        }
        results_file << "\n";
    }*/
    results_file.close();
}

void run_and_save_training(int batch_size, int epochs,
                           vector<vector<double>>& trainImages, vector<int>& trainLabels,
                           vector<vector<double>>& testImages, vector<int>& testLabels,
                           int input_length, int neurons, int num_outputs,
                           const vector<double>& chebyshev_relu_coefficients, const vector<double>& chebyshev_sigmoid_coefficients,
                           double chebyshev_a, double chebyshev_b,
                           double learning_rate = 0.001,
                           int num_networks = 10) {
    {
        lock_guard<mutex> guard(console_mutex);
        cout << "File opened" << endl;
        cout << "Current working directory: " << fs::current_path() << endl;
    }

    vector<thread> threads;
    for (int n = 0; n < num_networks; ++n) {
        threads.emplace_back(train_and_save, n, batch_size, epochs, ref(trainImages), ref(trainLabels),
                             ref(testImages), ref(testLabels), input_length, neurons, num_outputs,
                             ref(chebyshev_relu_coefficients), ref(chebyshev_sigmoid_coefficients),
                             chebyshev_a, chebyshev_b, learning_rate, num_networks);
    }

    for (auto& th : threads) {
        th.join();
    }
}


int main() {
    int neurons = 512;
    int input_length = 784;
    int batch_size = 32;
    int epochs = 15;
    const int degree = 10;
    const double chebyshev_a = -60.0;
    const double chebyshev_b = 60.0;
    const int num_outputs = 10;
    const int num_networks = 10;
    const double learning_rate = 0.001;

    // Generate ReLU Chebyshev coefficients
    vector<double> chebyshev_relu_coefficients = generate_chebyshev_coefficients_relu(degree, chebyshev_a, chebyshev_b);

    // Generate Sigmoid Chebyshev coefficients
    vector<double> chebyshev_sigmoid_coefficients = generate_chebyshev_coefficients_sigmoid(degree, chebyshev_a, chebyshev_b);

    {
        lock_guard<mutex> guard(console_mutex);
        cout << "Variables initialized" << endl;
    }

    // Load MNIST data
    string trainImagesPath = "/Users/Smaran/CLionProjects/FinalProjectFeedForwardNeuralNetwork/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte";
    string trainLabelsPath = "/Users/Smaran/CLionProjects/FinalProjectFeedForwardNeuralNetwork/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte";
    int numberOfImages;
    int imageSize;
    vector<vector<double>> trainImages = readMNISTImages(trainImagesPath, numberOfImages, imageSize);
    int numberOfLabels;
    vector<int> trainLabels = readMNISTLabels(trainLabelsPath, numberOfLabels);

    if (numberOfImages != numberOfLabels) {
        cerr << "Number of images and labels do not match!" << endl;
        exit(1);
    }

    string testImagesPath = "/Users/Smaran/CLionProjects/FinalProjectFeedForwardNeuralNetwork/MNIST/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";
    string testLabelsPath = "/Users/Smaran/CLionProjects/FinalProjectFeedForwardNeuralNetwork/MNIST/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";
    int numberOfTestImages;
    vector<vector<double>> testImages = readMNISTImages(testImagesPath, numberOfTestImages, imageSize);
    int numberOfTestLabels;
    vector<int> testLabels = readMNISTLabels(testLabelsPath, numberOfTestLabels);

    if (numberOfTestImages != numberOfTestLabels) {
        cerr << "Number of images and labels do not match!" << endl;
        exit(1);
    }

    {
        lock_guard<mutex> guard(console_mutex);
        cout << "MNIST data loaded" << endl;
    }

    run_and_save_training(batch_size, epochs, trainImages, trainLabels, testImages, testLabels, input_length, neurons, num_outputs, chebyshev_relu_coefficients, chebyshev_sigmoid_coefficients, chebyshev_a, chebyshev_b, learning_rate, num_networks);

    return 0;
}