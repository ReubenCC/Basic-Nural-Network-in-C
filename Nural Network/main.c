#define uint unsigned int

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>

typedef struct {
	uint size;
	float* values;
	float* biases; 
	float** weights; // hight = size, width = previose size
}Layer;

typedef struct {
	uint num_inputs;
	uint size; // number of layers not including the input layer
	float* inputs;
	Layer** layers;
	float* outputs;
}Network;

Layer* new_layer(uint num_inputs, uint num_nurons) {
	Layer* layer = malloc(sizeof*layer);
	layer->size = num_nurons;

	layer->values = malloc(sizeof(float) * num_nurons);
	layer->biases = malloc(sizeof(float) * num_nurons);
	memset(layer->biases, 0, sizeof(float) * num_nurons);

	layer->weights = malloc(sizeof(float*) * num_nurons);
	// he weight initualisation using polar box millar distrabution
	float stdDev = sqrt(2.0 / num_inputs);
	for (uint i = 0; i < num_nurons; i++){
		layer->weights[i] = malloc(sizeof(float) * num_inputs);
		for (uint ii = 0; ii < num_inputs; ii++) {
			float u, v, s;
			do {
				u = 2 * ((double)rand()) / RAND_MAX - 1;
				v = 2 * ((double)rand()) / RAND_MAX - 1;
				s = u * u + v * v;
			} while (s >= 1 || s == 0);
			s = stdDev * sqrt(-2.0 * log(s) / s);

			layer->weights[i][ii] = u * s;
			ii++;
			if (ii < num_inputs)
				layer->weights[i][ii] = v * s;
		}
	}

	return layer;
}

Network* new_network(uint size, uint* dimentions) {
	Network* network = malloc(sizeof*network);
	network->num_inputs = *dimentions;
	network->inputs = malloc(*dimentions * sizeof(float));
	network->outputs = malloc(dimentions[size-2] * sizeof(float));
	network->size = size-1;
	network->layers = malloc(sizeof(Layer*)*(size-1));
	for (uint i = 1; i < size; i++)
		network->layers[i-1] = new_layer(dimentions[i-1], dimentions[i]);

	return network;
};

float activation(float input) {
	// rectified liniar
	return (input < 0) ? 0 : input;
}

void foward_hidden_layer(Layer* layer, float* inputs, uint num_inputs) {
	memset(layer->values, 0, sizeof(float)*layer->size);

	for (uint i = 0; i < layer->size; i++) {
		for (uint ii = 0; ii < num_inputs; ii++)
			layer->values[i] += layer->weights[i][ii] * inputs[ii];

		layer->values[i] += layer->biases[i];

		layer->values[i] = activation(layer->values[i]);
	}
}

void foward_output_layer(Layer* layer, float* inputs, float* outputs, uint num_inputs) {
	memset(layer->values, 0, sizeof(float) * layer->size);

	float max_activation = -INFINITY;
	float sum = 0;

	for (uint i = 0; i < layer->size; i++) {
		for (uint ii = 0; ii < num_inputs; ii++)
			layer->values[i] += layer->weights[i][ii] * inputs[ii];

		layer->values[i] += layer->biases[i];

		if (layer->values[i] > max_activation)
			max_activation = layer->values[i];
	}
	// softmax
	for (uint i = 0; i < layer->size; i++) {
		layer->values[i] -= max_activation;
		outputs[i] = exp(layer->values[i]);
		sum += outputs[i];
	}
	for (uint i = 0; i < layer->size; i++)
		outputs[i] /= sum;
}

void process_network(Network* network) {
	foward_hidden_layer(network->layers[0], network->inputs, network->num_inputs);

	for (uint i = 1; i < network->size - 1; i++)
		foward_hidden_layer(network->layers[i], network->layers[i-1]->values, network->layers[i-1]->size);

	foward_output_layer(network->layers[network->size - 1], network->layers[network->size - 2]->values, network->outputs, network->layers[network->size - 2]->size);
}

void test_network(Network* network) {
	FILE* testing_lables = fopen("test-labels", "rb");
	FILE* testing_images = fopen("test-images", "rb");

	fseek(testing_lables, 8, SEEK_SET);
	fseek(testing_images, 16, SEEK_SET);
	double acuracy = 0;
	for (int i = 0; i < 10000; i++) {
		unsigned char lable;
		unsigned char raw_img[784];
		fread(&lable, 1, 1, testing_lables);
		fread(raw_img, 1, 784, testing_images);
		for (short i = 0; i < 784; i++)
			network->inputs[i] = (float)raw_img[i] / 255;
		process_network(network);
		int max = 0;
		for (char i = 1; i < 10; i++) {
			if (network->outputs[i] > network->outputs[max])
				max = i;
		}
		if (max == lable)
			acuracy++;
	}
	fclose(testing_lables);
	fclose(testing_images);

	acuracy /= 100;
	printf("%3.2f%%\n", acuracy);
}

void calculate_gradient(Network* network, float*** weights, float** biases, unsigned char lable) {
	float* outputs = malloc(sizeof(float) * 10); // optimise this out
	memset(outputs, 0, sizeof(float) * 10);
	for (char i = 0; i < 10; i++) {
		for (uint ii = 0; ii < network->layers[network->size-2]->size; ii++)
			outputs[i] += network->layers[network->size-2]->values[ii] * network->layers[network->size-1]->weights[i][ii];
		outputs[i] += network->layers[network->size-1]->biases[i];
	}

	char max = 9;
	float sum = 0;
	for (char i = 0; i < 9; i++) {
		if (outputs[i] > outputs[max]) {
			max = i;
		}
	}
	for (char i = 0; i < 10; i++) {
		sum += exp(outputs[i] - outputs[max]);
	}

	for (unsigned char i = 0; i < 10; i++) {
		biases[network->size-1][i] = exp(outputs[i]-outputs[max]) / sum - (i == lable);
		for (uint ii = 0; ii < network->layers[network->size-2]->size; ii++)
			weights[network->size-1][i][ii] += biases[network->size-1][i] * network->layers[network->size-2]->values[ii];
	}
	for (long i = network->size - 2; i >= 0; i--) {
		memset(biases[i], 0, sizeof(float) * network->layers[i]->size);
		for (uint ii = 0; ii < network->layers[i]->size; ii++) {
			for (uint iii = 0; iii < network->layers[i+1]->size; iii++)
				biases[i][ii] += biases[i+1][iii] * network->layers[i+1]->weights[iii][ii];
			biases[i][ii] *= network->layers[i]->values[ii] != 0;

			for (uint iii = 0; iii < ((i)?network->layers[i-1]->size:network->num_inputs); iii++)
				weights[i][ii][iii] += biases[i][ii] * ((i)?network->layers[i-1]->values[iii]:network->inputs[iii]);
		}
	}
	free(outputs);
}

void train_network(Network* network, uint batch_size, float learning_rate) {
	FILE* training_lables = fopen("training-labels", "rb");
	FILE* training_images = fopen("training-images", "rb");

	fseek(training_lables, 8, SEEK_SET);
	fseek(training_images, 16, SEEK_SET);

	float*** weights = malloc(sizeof(float**) * network->size);
	float*** biases = malloc(sizeof(float**) * batch_size);

	for (uint i = 0; i < network->size; i++) {
		weights[i] = malloc(sizeof(float*) * network->layers[i]->size);
		for (uint ii = 0; ii < network->layers[i]->size; ii++)
			weights[i][ii] = malloc(sizeof(float) * ((i)?(network->layers[i-1]->size):(network->num_inputs)));
	}

	for (char i = 0; i < batch_size; i++) {
		biases[i] = malloc(sizeof(float*) * network->size);
		for (uint ii = 0; ii < network->size; ii++)
			biases[i][ii] = malloc(sizeof(float) * network->layers[ii]->size);
	}

	for (short i = 0; i < 3750; i++) {
		if (!(i%375))
			test_network(network);
		
		for (uint ii = 0; ii < network->size; ii++)
			for (uint iii = 0; iii < network->layers[ii]->size; iii++)
				memset(weights[ii][iii], 0, sizeof(float)*((ii)?(network->layers[ii-1]->size):(network->num_inputs)));

		for (char ii = 0; ii < batch_size; ii++) {
			unsigned char lable;
			unsigned char raw_img[784];
			fread(&lable, 1, 1, training_lables);
			fread(raw_img, 1, 784, training_images);

			for (short iii = 0; iii < 784; iii++)
				network->inputs[iii] = (float)raw_img[iii] / 255;

			process_network(network);

			calculate_gradient(network, weights, biases[ii], lable);
		}
		for (char ii = 1; ii < batch_size; ii++)
			for (uint iii = 0; iii < network->size; iii++)
				for (uint iv = 0; iv < network->layers[iii]->size; iv++)
					biases[0][iii][iv] += biases[ii][iii][iv];

		for (uint ii = 0; ii < network->size; ii++) {
			for (uint iii = 0; iii < network->layers[ii]->size; iii++) {
				network->layers[ii]->biases[iii] -= biases[0][ii][iii] / batch_size * learning_rate;

				for (uint iv = 0; iv < ((ii)?(network->layers[ii-1]->size):(network->num_inputs)); iv++)
					network->layers[ii]->weights[iii][iv] -= weights[ii][iii][iv] / batch_size * learning_rate;
			}
		}
	}

	for (uint i = 0; i < network->size; i++) {
		for (uint ii = 0; ii < network->layers[i]->size; ii++)
			free(weights[i][ii]);
		free(weights[i]);
	}
	free(weights);
	for (char i = 0; i < 16; i++) {
		for (uint ii = 0; ii < network->size; ii++)
			free(biases[i][ii]);
		free(biases[i]);
	}
	free(biases);
	fclose(training_lables);
	fclose(training_images);
}

int main(int argc, char** argv) {
	srand(time(0));
	//
	//uint* dimentions = malloc(sizeof(uint) * 3);
	//dimentions[0] = 2;
	//dimentions[1] = 3;
	//dimentions[2] = 2;
	//
	//Network* network = new_network(3, dimentions);
	//network->inputs[0] = 1;
	//network->inputs[1] = 1;
	//
	//process_network(network);
	//
	//for (uint i = 0; i < network->layers[network->size - 1]->size; i++)
	//	printf("%f\n\n", network->layers[network->size - 1]->values[i]);
	//for (uint i = 0; i < network->layers[0]->size; i++)
	//	for (uint ii = 0; ii < network->num_inputs; ii++)
	//		printf("%f\n", network->layers[0]->weights[i][ii]);
	uint* dimentions = malloc(sizeof(uint) * 4);
	dimentions[0] = 784;
	dimentions[1] = 200;
	dimentions[2] = 80;
	dimentions[3] = 10;

	Network* network = new_network(4, dimentions);

	train_network(network, 16, 0.2);

	test_network(network);
}