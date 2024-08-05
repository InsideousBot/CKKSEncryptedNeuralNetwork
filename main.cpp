//
// Created by Smaran Manchala on 8/2/24.
//
#include <iostream>
#include <fstream>
#include "Operations.h"
#include "MNISTReader.h"
#include <chrono>
#include <filesystem>

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
                                        const vector<double>& chebyshev_coefficients_derivative, double chebyshev_a, double chebyshev_b) {
    vector<vector<double>> sigmoid_derivative = chebyshev_sigmoid_derivative_approx(inputs, chebyshev_coefficients_derivative, chebyshev_a, chebyshev_b);
    vector<vector<double>> dinputs(dvalues.size(), vector<double>(dvalues[0].size(), 0.0));

    for (size_t i = 0; i < dinputs.size(); ++i) {
        for (size_t j = 0; j < dinputs[0].size(); ++j) {
            dinputs[i][j] = dvalues[i][j] * sigmoid_derivative[i][j];
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
                          const vector<double>& chebyshev_coefficients, double chebyshev_a, double chebyshev_b,
                          const vector<vector<double>>& testImages, const vector<int>& testLabels) {
    // Forward pass through the hidden layer
    vector<vector<double>> hidden_layer_outputs = forward_hidden_layer(testImages, weights1, bias1);

    // Apply the sigmoid activation function
    vector<vector<double>> sigmoid_outputs = forward_sigmoid(hidden_layer_outputs, chebyshev_coefficients, chebyshev_a, chebyshev_b);

    // Forward pass through the output layer
    vector<vector<double>> output_layer_outputs = forward_output_layer(sigmoid_outputs, weights2, bias2);

    // Apply the softmax function
    vector<vector<double>> softmax_outputs = forward_softmax(output_layer_outputs);

    // Calculate the loss
    vector<double> sample_losses = cross_entropy_loss_forward(softmax_outputs, testLabels);
    double test_loss = calculate_mean_loss(sample_losses);

    // Calculate accuracy
    double test_accuracy = calculate_accuracy(softmax_outputs, testLabels);

    return make_pair(test_loss, test_accuracy);
}

vector<vector<double>> train(int batch_size, int epochs,
                             vector<vector<double>>& trainImages, vector<int>& trainLabels, // Added trainImages and trainLabels as parameters
                             vector<vector<double>>& testImages, vector<int>& testLabels,  // Added testImages and testLabels as parameters
                             int input_length, int neurons, int num_outputs,
                             const vector<double>& chebyshev_coefficients, const vector<double>& chebyshev_derivative_coefficients,
                             double chebyshev_a, double chebyshev_b,
                             double learning_rate = 0.001) {
    int numberOfImages = trainImages.size();
    cout << "Loop starting" << endl;

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Reinitialize weights and biases at the start of each epoch
        vector<vector<double>> weights1 = generateRandomMatrix(input_length, neurons, 0.0, 1.0);
        vector<vector<double>> bias1 = generateZeroMatrix(1, neurons);
        vector<vector<double>> weights2 = generateRandomMatrix(neurons, num_outputs, 0.0, 1.0);
        vector<vector<double>> bias2 = generateZeroMatrix(1, num_outputs);

        double epoch_loss = 0.0;
        double epoch_accuracy = 0.0;
        cout << "Epoch: " << epoch << endl;
        for (int i = 0; i < numberOfImages; i += batch_size) {
            // Prepare batch
            vector<vector<double>> inputs(trainImages.begin() + i, trainImages.begin() + min(i + batch_size, numberOfImages));
            vector<int> batch_labels(trainLabels.begin() + i, trainLabels.begin() + min(i + batch_size, numberOfImages));

            // Forward pass
            vector<vector<double>> hidden_layer_outputs = forward_hidden_layer(inputs, weights1, bias1);
            vector<vector<double>> sigmoid_outputs = forward_sigmoid(hidden_layer_outputs, chebyshev_coefficients, chebyshev_a, chebyshev_b);
            vector<vector<double>> output_layer_outputs = forward_output_layer(sigmoid_outputs, weights2, bias2);
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
            vector<vector<double>> dinputs2 = backward_layer(dSoftmaxCrossEntropyInputs, sigmoid_outputs, weights2, dweights2, dbiases2);

            vector<vector<double>> dSigmoidInputs = backward_sigmoid(dinputs2, hidden_layer_outputs, chebyshev_derivative_coefficients, chebyshev_a, chebyshev_b);

            vector<vector<double>> dweights1(weights1.size(), vector<double>(weights1[0].size(), 0.0));
            vector<vector<double>> dbiases1(1, vector<double>(bias1[0].size(), 0.0));
            vector<vector<double>> dinputs1 = backward_layer(dSigmoidInputs, inputs, weights1, dweights1, dbiases1);

            update_params(weights2, dweights2, bias2, dbiases2, learning_rate);
            update_params(weights1, dweights1, bias1, dbiases1, learning_rate);
        }
        epoch_loss /= (numberOfImages / batch_size);
        epoch_accuracy /= (numberOfImages / batch_size);
        cout << "Epoch: " << epoch << ", Loss: " << epoch_loss << ", Accuracy: " << epoch_accuracy << endl;

        // Test the model
        double test_loss, test_accuracy;
        tie(test_loss, test_accuracy) = test(weights1, bias1, weights2, bias2, chebyshev_coefficients, chebyshev_a, chebyshev_b, testImages, testLabels);
        cout << "Test Loss: " << test_loss << ", Test Accuracy: " << test_accuracy << endl;
    }
    return {};
}

void run_and_save_training(int batch_size, int epochs,
                           vector<vector<double>>& trainImages, vector<int>& trainLabels,
                           vector<vector<double>>& testImages, vector<int>& testLabels,
                           int input_length, int neurons, int num_outputs,
                           const vector<double>& chebyshev_coefficients, const vector<double>& chebyshev_derivative_coefficients,
                           double chebyshev_a, double chebyshev_b,
                           double learning_rate = 0.001) {
    cout << "File opened" << endl;
    ofstream results_file("training_results.txt");
    cout << "Current working directory: " << fs::current_path() << endl;
    if(results_file.is_open())
    {
        results_file << "Training Results:\n";
    }
    else
    {
        cerr << "Unable to open file for writing\n";
    }
    cout << "Clock starting" << endl;
    auto start_time = high_resolution_clock::now();
    // Initialize weights and biases for hidden layer
    vector<vector<double>> weights1 = generateRandomMatrix(input_length, neurons, 0.0, 1.0);
    vector<vector<double>> bias1 = generateZeroMatrix(1, neurons);

    // Initialize weights and biases for output layer
    vector<vector<double>> weights2 = generateRandomMatrix(neurons, num_outputs, 0.0, 1.0);
    vector<vector<double>> bias2 = generateZeroMatrix(1, num_outputs);

    // Train the network
    vector<vector<double>> result = train(batch_size, epochs, trainImages, trainLabels, testImages, testLabels, input_length, neurons, num_outputs, chebyshev_coefficients, chebyshev_derivative_coefficients, chebyshev_a, chebyshev_b, learning_rate);

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(end_time - start_time).count();

    // Save the results to a file
    if (results_file.is_open()) {
        results_file << "Training Time: " << duration << " seconds\n";
        results_file << "Epochs: " << epochs << "\n";
        results_file << "Batch Size: " << batch_size << "\n";
        results_file << "Learning Rate: " << learning_rate << "\n";
        // Save final weights and biases
        results_file << "Weights1:\n";
        for (const auto& row : weights1) {
            for (const auto& val : row) {
                results_file << val << " ";
            }
            results_file << "\n";
        }
        results_file << "Bias1:\n";
        for (const auto& row : bias1) {
            for (const auto& val : row) {
                results_file << val << " ";
            }
            results_file << "\n";
        }
        results_file << "Weights2:\n";
        for (const auto& row : weights2) {
            for (const auto& val : row) {
                results_file << val << " ";
            }
            results_file << "\n";
        }
        results_file << "Bias2:\n";
        for (const auto& row : bias2) {
            for (const auto& val : row) {
                results_file << val << " ";
            }
            results_file << "\n";
        }
        results_file << "Accuracy: " << calculate_accuracy(result, testLabels) << "\n";
        results_file.close();
    } else {
        cerr << "Unable to open file for writing\n";
    }
}

int main() {
    int neurons = 512;
    int input_length = 784;
    int batch_size = 32;
    int epochs = 5;
    const int degree = 10;
    const double chebyshev_a = -60.0;
    const double chebyshev_b = 60.0;
    const int num_outputs = 10;

    // Generate Chebyshev coefficients
    vector<double> chebyshev_coefficients = generate_chebyshev_coefficients_sigmoid(degree, chebyshev_a, chebyshev_b);
    vector<double> chebyshev_derivative_coefficients = generate_chebyshev_coefficients_sigmoid_derivative(degree, chebyshev_a, chebyshev_b);

    cout << "Variables initialized" << endl;

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

    cout << "MNIST data loaded" << endl;


    run_and_save_training(batch_size, epochs, trainImages, trainLabels, testImages, testLabels, input_length, neurons, num_outputs, chebyshev_coefficients, chebyshev_derivative_coefficients, chebyshev_a, chebyshev_b);
    return 0;
}

vector<vector<double>> backward_relu(const vector<vector<double>>& dvalues, const vector<vector<double>>& inputs) {
    vector<vector<double>> dinputs = dvalues; // Copy dvalues

    // Zero gradient where input values were negative or zero
    for (size_t i = 0; i < dinputs.size(); ++i) {
        for (size_t j = 0; j < dinputs[0].size(); ++j) {
            if (inputs[i][j] <= 0) {
                dinputs[i][j] = 0;
            }
        }
    }
    cout << "dInputs size relu: " << dinputs.size() << " x " << dinputs[0].size() << endl;
    return dinputs;
}
