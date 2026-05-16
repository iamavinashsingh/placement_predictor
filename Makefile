# Makefile for Placement Predictor

CXX = g++
CXXFLAGS = -O3 -Wall

all: train predict

train: placement_nn.cpp nn_lib.h
	$(CXX) $(CXXFLAGS) placement_nn.cpp -o train

predict: predict.cpp nn_lib.h
	$(CXX) $(CXXFLAGS) predict.cpp -o predict

clean:
	rm -f train predict model.txt
