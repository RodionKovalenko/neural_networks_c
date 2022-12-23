/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.c to edit this template
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <string.h>
#include "../layer.h"
#include "../feedforward_network.h"
#include "../weight_initializer.h"
#include "../verbose.h"
#include "../array.h"
#include "../math.h"

static int verbose = 0;

void set_verbose(int value) {
    verbose = value;
}

feedforward_network init_ffn(
        double **inputs,
        int *input_dims,
        int input_num_records,
        int num_input_params,
        int n_h_layers,
        int n_h_neurons,
        int n_out_neurons,
        double **targets,
        double learning_rate
        ) {

    layer *layers = init_layers(inputs, input_dims, num_input_params, n_h_layers, n_h_neurons, n_out_neurons);

    feedforward_network ffn = {
        .num_records = input_num_records,
        .dataset = inputs,
        .learning_rate = learning_rate,
        .n_h_layers = n_h_layers,
        .n_h_neurons = n_h_neurons,
        .n_input_rows = layers[0].num_input_rows,
        .n_input_columns = layers[0].num_input_rows,
        .n_out_neurons = n_out_neurons,
        .input_dims = input_dims,
        .targets = targets,
        .layers = layers
    };

    if (verbose == 1) {
        print_network(ffn);
    }

    return ffn;
}

struct layer* init_layers(
        double **inputs,
        int *input_dims,
        int num_input_params,
        int n_h_layers,
        int n_h_neurons,
        int n_out_neurons
        ) {
    int i;

    layer *layers = (layer *) malloc(sizeof (struct layer) * (n_h_layers + 2));

    layer *input_layer = (layer *) malloc(sizeof (struct layer));

    input_layer->layer_name = "input layer";
    layers[0] = *input_layer;
    layers[0].inputs = inputs;
    layers[0].outputs = inputs;

    // input layer 0
    for (i = 0; i < num_input_params; i++) {
        if (i == 1) {
            layers[0].num_inputs = input_dims[i];
            layers[0].num_input_rows = input_dims[i];
        }
        if (i == 2) {
            layers[0].num_outputs = input_dims[i];
            layers[0].num_input_columns = input_dims[i];
        }
    }
    layers[0].layer_index = 1;


    // hidden layers
    for (i = 1; i < n_h_layers + 1; i++) {
        layer *hidden_layer = (layer *) malloc(sizeof (struct layer));
        hidden_layer->num_inputs = layers[i - 1].num_outputs;
        hidden_layer->num_outputs = n_h_neurons;

        hidden_layer->num_input_rows = layers[i - 1].num_outputs;
        hidden_layer->num_input_columns = n_h_neurons;

        double **weight_matrix = build_array(hidden_layer->num_outputs, hidden_layer->num_inputs);

        init_random_weights(weight_matrix, hidden_layer->num_outputs, hidden_layer->num_inputs);
        char *layer_name = "hidden layer";

        hidden_layer->layer_name = layer_name;
        hidden_layer->layer_index = (i + 1);
        hidden_layer->weights = weight_matrix;

        hidden_layer->outputs = build_array(hidden_layer->num_outputs, input_dims[1]);
        hidden_layer->previous_layer = &layers[i - 1];
        hidden_layer->next_layer = &layers[i + 1];
        layers[i] = *hidden_layer;
        layers[i - 1].next_layer = hidden_layer;
    }

    // output layer
    layer *output_layer = (layer *) malloc(sizeof (struct layer));
    double **weight_matrix = build_array(n_out_neurons, n_h_neurons);

    init_random_weights(weight_matrix, n_out_neurons, n_h_neurons);

    output_layer->weights = weight_matrix;

    output_layer->num_input_rows = layers[i - 1].num_outputs;
    output_layer->num_input_columns = n_out_neurons;
    output_layer->num_inputs = layers[i - 1].num_inputs;
    output_layer->num_outputs = n_out_neurons;
    output_layer->layer_name = "output layer";
    output_layer->layer_index = (i + 1);
    output_layer->previous_layer = &layers[i - 1];
    output_layer->outputs = build_array(output_layer->num_outputs, input_dims[1]);

    layers[i] = *output_layer;
    layers[i - 1].next_layer = output_layer;

    return layers;
}

void forward(feedforward_network ffn, int record_index) {
    int i, j, r, l, k;

    for (l = 1; l < ffn.n_h_layers + 2; l++) {
        layer *_layer = &ffn.layers[l];
        layer *_prev_layer = _layer->previous_layer;

        if (_prev_layer != NULL) {
            double **layer_input = _prev_layer->outputs;
            double **outputs = _layer->outputs;

            clear_array(outputs, _layer->num_outputs, ffn.input_dims[1]);

            for (j = 0; j < _layer->num_outputs; j++) {
                for (i = 0; i < _layer->num_inputs; i++) {
                    for (k = 0; k < ffn.input_dims[1]; k++) {
                        if (layer_input != NULL) {
                            if (_prev_layer->layer_index == 1) {
                                outputs[j][k] += _layer->weights[j][i] * ffn.dataset[record_index][i];
                            } else {
                                outputs[j][k] += _layer->weights[j][i] * layer_input[i][k];
                            }
                        }
                    }
                }
            }

            sigmoid_to_matrix(outputs, _layer->num_outputs, ffn.input_dims[1]);
            _layer->outputs = outputs;

            if (_prev_layer != NULL && outputs != NULL && verbose == 1) {
                printf("number of input dimensions/columns %d \n", ffn.input_dims[2]);
                for (i = 0; i < ffn.input_dims[2]; i++) {
                    printf("record data are: column %d, value: %f \n", i, ffn.dataset[record_index][i]);
                }

                printf("output dimensions: %d x %d x %d \n\n\n", ffn.num_records, _layer->num_outputs, ffn.input_dims[1]);
                printf("layer index %d \n", _layer->layer_index);
                printf("record index %d \n", record_index);
                print_matrix_double(outputs, _layer->num_outputs, ffn.input_dims[1]);
            }
        }
    }
}

feedforward_network fit(feedforward_network ffn, int num_iterations, int training_mode) {
    int i, j, record_index;

    for (i = 0; i < num_iterations; i++) {
        for (record_index = 0; record_index < ffn.num_records; record_index++) {
            forward(ffn, record_index);
        }
    }
}