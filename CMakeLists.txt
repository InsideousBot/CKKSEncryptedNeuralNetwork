cmake_minimum_required(VERSION 3.27)
project(FinalProjectFeedForwardNeuralNetwork)

set(CMAKE_CXX_STANDARD 17)

add_executable(FinalProjectFeedForwardNeuralNetwork main.cpp
        Operations.h
        Operations.cpp
        MNISTReader.h
        MNISTReader.cpp)
