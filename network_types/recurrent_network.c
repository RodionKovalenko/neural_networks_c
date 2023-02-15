/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.c to edit this template
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "layer.h"
#include "network.h"
#include "recurrent_network.h"
#include "../utils/weight_initializer.h"
#include "../utils/verbose.h"
#include "../utils/array.h"
#include "../utils/math.h"
#include "../utils/activation.h"
#include "../utils/loss_function.h"
#include "../utils/optimizer.h"

static int verbose = 0;

struct layer* init_rnn_layers(
        int *input_dims,
        int num_input_params,
        int n_h_layers,
        int n_h_neurons,
        int n_out_neurons,
        int activation,
        double bottleneck_value,
        int batch_size
        ) {
    int i;

    layer *layers = (layer *) malloc(sizeof (struct layer) * (n_h_layers + 2));
    layer *input_layer = (layer *) malloc(sizeof (struct layer));

    input_layer->layer_name = "input layer";
    layers[0] = *input_layer;

    // input layer 0
    for (i = 0; i < num_input_params; i++) {
        if (i == 2) {
            layers[0].num_inputs = input_dims[i];
            layers[0].num_input_rows = input_dims[i];
            layers[0].num_outputs = input_dims[i];
            layers[0].num_input_columns = input_dims[i];
        }
    }
    layers[0].layer_index = 1;

    int m = n_h_layers + 2;
    int k;

    // hidden layers
    for (i = 1; i < n_h_layers + 1; i++) {
        layer *hidden_layer = (layer *) malloc(sizeof (struct layer));
        hidden_layer->num_inputs = layers[i - 1].num_outputs;
        hidden_layer->num_outputs = n_h_neurons;

        hidden_layer->num_input_rows = layers[i - 1].num_outputs;
        hidden_layer->num_input_columns = n_h_neurons;
        k = i + 1;

        // bottleneck of hidden layers 
        if (k > 2 && bottleneck_value != 0) {
            if ((m - k + 1) >= k) {
                hidden_layer->num_inputs = layers[i - 1].num_outputs;
                hidden_layer->num_outputs = (int) ((double) layers[i - 1].num_outputs / bottleneck_value) + 1;

                hidden_layer->num_input_rows = layers[i - 1].num_outputs;
                hidden_layer->num_input_columns = (int) ((double) layers[i - 1].num_outputs / bottleneck_value) + 1;
            } else {
                hidden_layer->num_inputs = layers[i - 1].num_outputs;
                hidden_layer->num_outputs = layers[m - k + 1].num_inputs;

                hidden_layer->num_input_rows = layers[i - 1].num_outputs;
                hidden_layer->num_input_columns = layers[m - k + 1].num_inputs;
            }
        }

        double **weight_matrix = build_array(hidden_layer->num_outputs, hidden_layer->num_inputs);
        double **rnn_prev_weight_matrix = build_array(hidden_layer->num_outputs, hidden_layer->num_outputs);

        init_random_weights(weight_matrix, hidden_layer->num_outputs, hidden_layer->num_inputs);
        init_random_weights(rnn_prev_weight_matrix, hidden_layer->num_outputs, hidden_layer->num_outputs);

        char *layer_name = "hidden layer";

        hidden_layer->adam_A = build_array(hidden_layer->num_outputs, hidden_layer->num_inputs);
        hidden_layer->adam_B = build_array(hidden_layer->num_outputs, hidden_layer->num_inputs);
        hidden_layer->adam_A_bias = build_array(hidden_layer->num_outputs, input_dims[1]);
        hidden_layer->adam_B_bias = build_array(hidden_layer->num_outputs, input_dims[1]);

        hidden_layer->layer_name = layer_name;
        hidden_layer->layer_index = k;
        hidden_layer->weights = weight_matrix;
        hidden_layer->prev_layer_weights = rnn_prev_weight_matrix;

        hidden_layer->outputs = build_array(hidden_layer->num_outputs, input_dims[1]);
        hidden_layer->layer_prev_outputs = build_array_3d(batch_size, hidden_layer->num_outputs, input_dims[1]);
        hidden_layer->bias = build_array(hidden_layer->num_outputs, 1);
        hidden_layer->gradients = build_array(hidden_layer->num_outputs, input_dims[1]);
        hidden_layer->gradients_B = build_array(hidden_layer->num_outputs, 1);
        hidden_layer->gradients_W = build_array(hidden_layer->num_outputs, hidden_layer->num_inputs);
        hidden_layer->activation_type = activation;

        hidden_layer->previous_layer = &layers[i - 1];
        hidden_layer->next_layer = &layers[k];
        layers[i] = *hidden_layer;
        layers[i - 1].next_layer = hidden_layer;
    }

    if (n_h_layers == 0) {
        n_h_neurons = input_dims[2];
    }

    // output layer
    layer *output_layer = (layer *) malloc(sizeof (struct layer));

    double **weight_matrix = build_array(n_out_neurons, n_h_neurons);
    init_random_weights(weight_matrix, n_out_neurons, n_h_neurons);

    output_layer->weights = weight_matrix;

    output_layer->num_input_rows = layers[i - 1].num_outputs;
    output_layer->num_input_columns = n_out_neurons;
    output_layer->num_inputs = layers[i - 1].num_outputs;
    output_layer->num_outputs = n_out_neurons;
    output_layer->layer_name = "output layer";
    output_layer->layer_index = (i + 1);
    output_layer->previous_layer = &layers[i - 1];
    output_layer->outputs = build_array(output_layer->num_outputs, input_dims[1]);
    output_layer->bias = build_array(output_layer->num_outputs, 1);
    output_layer->gradients = build_array(output_layer->num_outputs, input_dims[1]);
    output_layer->gradients_B = build_array(output_layer->num_outputs, 1);
    output_layer->gradients_W = build_array(output_layer->num_outputs, output_layer->num_inputs);

    output_layer->adam_A = build_array(output_layer->num_outputs, output_layer->num_inputs);
    output_layer->adam_B = build_array(output_layer->num_outputs, output_layer->num_inputs);
    output_layer->adam_A_bias = build_array(output_layer->num_outputs, input_dims[1]);
    output_layer->adam_B_bias = build_array(output_layer->num_outputs, input_dims[1]);

    output_layer->activation_type = activation;

    layers[i] = *output_layer;
    layers[i - 1].next_layer = output_layer;

    return layers;
}

network init_rnn(
        int *input_dims,
        int num_input_params,
        int input_num_records,
        int n_h_layers,
        int n_h_neurons,
        int n_out_neurons,
        double learning_rate,
        int activation,
        double bottleneck_value,
        int batch_size
        ) {

    // make n_h_layers odd if it is even
    if (n_h_layers % 2 == 0) {
        n_h_layers += 1;
    }

    layer *layers = init_rnn_layers(input_dims, num_input_params, n_h_layers, n_h_neurons, n_out_neurons, activation, bottleneck_value, batch_size);

    network ffn = {
        .num_records = input_num_records,
        .learning_rate = learning_rate,
        .n_h_layers = n_h_layers,
        .n_h_neurons = n_h_neurons,
        .n_out_neurons = n_out_neurons,
        .minibatch_size = 15,
        .input_dims = input_dims,
        .layers = layers,
        .activation_type = activation,
        .errors = build_array(n_out_neurons, input_dims[1]),
        .is_gradient_checked = 0,
        .loss_function = MEAN_SQUARED_ERROR_LOSS,
        .optimizer = DEFAULT,
    };

    printf("layer features %d \n", ffn.n_features);

    ffn.layers[0].outputs = build_array(1, ffn.n_features);

    if (verbose == 1) {
        print_network(ffn);
    }

    return ffn;
}

void forward_rnn(network ffn, double **data_X) {
    int d, l;
    double **layer_input, **outputs;
    layer *_layer, *_prev_layer;

    for (d = 0; d < ffn.batch_size; d++) {
        for (l = 1; l < ffn.n_h_layers + 2; l++) {
            _layer = &ffn.layers[l];
            _prev_layer = _layer->previous_layer;

            if (l == 1) {
                _prev_layer->outputs[0] = data_X[d];
            }

            if (_prev_layer != NULL) {
                layer_input = _prev_layer->outputs;
                outputs = _layer->outputs;

                if (_prev_layer->layer_index == 1) {
                    outputs = apply_matrix_product_transposed(outputs, _layer->weights, layer_input, _layer->num_outputs, ffn.input_dims[1], _layer->num_inputs);
                } else {
                    outputs = apply_matrix_product(outputs, _layer->weights, layer_input, _layer->num_outputs, ffn.input_dims[1], _layer->num_inputs);
                }

                if (_layer->layer_prev_outputs != NULL && _layer->prev_layer_weights != NULL && d > 0) {
                    //                    printf("hidden d index %d\n", _layer->layer_index);
                    //                    print_matrix_double_3d(_layer->layer_prev_outputs, d, _layer->num_outputs, ffn.input_dims[1]);

                    _layer->layer_prev_outputs[d] = apply_matrix_product(_layer->layer_prev_outputs[d], _layer->prev_layer_weights, _layer->layer_prev_outputs[d - 1], _layer->num_outputs, ffn.input_dims[1], _layer->num_outputs);
                    outputs = matrix_add_matrix(_layer->layer_prev_outputs[d], outputs, _layer->num_outputs, ffn.input_dims[1]);
                }

                _layer->outputs = matrix_add_bias(outputs, _layer->bias, _layer->num_outputs, ffn.input_dims[1]);
                apply_activation(_layer, ffn);

                if (_layer->layer_prev_outputs != NULL) {
                    _layer->layer_prev_outputs[d] = copy_array(_layer->layer_prev_outputs[d], _layer->outputs, _layer->num_outputs, ffn.input_dims[1]);
                }

                if (l == ffn.n_h_layers + 1) {
                    printf("final layer index %d\n", _layer->layer_index);
                    print_matrix_double(_layer->outputs, _layer->num_outputs, ffn.input_dims[1]);
                }
            }
        }
    }
}

// https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

void backward_rnn(network ffn, double **data_X, double **target_Y) {
    //    int l;
    //    layer *_layer;
    //
    //    double **target_Y_matrix = convert_vector_to_matrix(target_Y, ffn.n_out_neurons, ffn.input_dims[1]);
    //    double **data_X_matrix = convert_vector_to_matrix(data_X, ffn.input_dims[2], ffn.input_dims[1]);
    //
    //    // calculate gradients
    //    for (l = ffn.n_h_layers + 1; l > 0; l--) {
    //        _layer = &ffn.layers[l];
    //
    //        if (l == (ffn.n_h_layers + 1)) {
    //            ffn.errors = get_errors_rnn(ffn, _layer->outputs, target_Y, _layer->num_outputs, ffn.input_dims[1]);
    //        }
    //
    //        _layer->gradients = calculate_jacobi_matrix_rnn(&ffn, _layer, target_Y_matrix, data_X_matrix);
    //    }
}

double **get_errors_rnn(network ffn, double **output, double **target_Y, int n_row, int n_col) {
    //    int i, j;
    //    int normalizing_constant = ffn.minibatch_size;
    //
    //    if (ffn.num_records < ffn.minibatch_size) {
    //        normalizing_constant = ffn.num_records;
    //    }
    //
    //    for (i = 0; i < n_row; i++) {
    //        for (j = 0; j < n_col; j++) {
    //            switch (ffn.loss_function) {
    //                case MEAN_SQUARED_ERROR_LOSS:
    //                    ffn.errors[j][i] += pow((target_Y[j] - output[i][j]), 2.0) / (double) normalizing_constant;
    //            }
    //        }
    //    }
    //
    return ffn.errors;
}

double **calculate_jacobi_matrix_rnn(network *ffn, layer *_layer, double **targetY, double **data_X) {
    //    int i, j, k, n;
    //    int n_row = _layer->num_outputs;
    //    int d = ffn->input_dims[1];
    //    layer *_next_layer = _layer->next_layer;
    //    layer *_prev_layer = _layer->previous_layer;
    //    double derivative;
    //
    //    for (i = 0; i < n_row; i++) {
    //        for (k = 0; k < d; k++) {
    //            derivative = apply_deactivation_to_value(_layer, i, k, *ffn);
    //
    //            if (_layer->layer_index == (ffn->n_h_layers + 2)) {
    //                _layer->gradients[i][k] += -2.0 * (targetY[i][k] - _layer->outputs[i][k]) * derivative;
    //            } else {
    //                for (n = 0; n < _next_layer->num_outputs; n++) {
    //                    _layer->gradients[i][k] += _next_layer->gradients[n][k] * _next_layer->weights[n][i] * derivative;
    //                }
    //            }
    //
    //            _layer->gradients_B[i][0] += _layer->gradients[i][k];
    //
    //            for (j = 0; j < _layer->num_inputs; j++) {
    //                if (_layer->layer_index == 2) {
    //                    _layer->gradients_W[i][j] += _layer->gradients[i][k] * data_X[j][k];
    //                } else {
    //                    _layer->gradients_W[i][j] += _layer->gradients[i][k] * _prev_layer->outputs[j][k];
    //                }
    //            }
    //        }
    //    }

    return _layer->gradients;
}

void update_weights_rnn(network ffn, int i_iteration) {
    int i, j, r, l, k;
    int normalizing_constant = ffn.minibatch_size;
    double p = 0.9999, pf = 0.9;
    double t = (double) i_iteration + 1;
    double prev_weight, prev_bias;
    double beta = 0.2;
    layer *_layer;

    if (ffn.num_records < ffn.minibatch_size) {
        normalizing_constant = ffn.num_records;
    }

    for (l = ffn.n_h_layers + 1; l >= 1; l--) {
        _layer = &ffn.layers[l];

        for (j = 0; j < _layer->num_outputs; j++) {
            for (i = 0; i < _layer->num_inputs; i++) {
                prev_weight = _layer->weights[j][i];
                switch (ffn.optimizer) {
                    case DEFAULT:
                        _layer->weights[j][i] -= ffn.learning_rate * _layer->gradients_W[j][i] / (double) normalizing_constant;
                        ffn.learning_rate *= 0.999;
                        break;
                    default:
                        _layer->weights[j][i] -= ffn.learning_rate * _layer->gradients_W[j][i] / (double) normalizing_constant;
                        break;
                }

                //_layer->weights[j][i] = beta * prev_weight + (1.0 - beta) * _layer->weights[j][i];
            }

            prev_bias = _layer->bias[j][0];

            switch (ffn.optimizer) {
                case DEFAULT:
                    _layer->bias[j][0] -= ffn.learning_rate * _layer->gradients_B[j][0] / (double) normalizing_constant;
                    break;
                default:
                    _layer->bias[j][0] -= ffn.learning_rate * _layer->gradients_B[j][0] / (double) normalizing_constant;
                    break;
            }

            _layer->bias[j][0] = beta * prev_bias + (1.0 - beta) * _layer->bias[j][0];
        }

        clear_array(_layer->gradients, _layer->num_outputs, ffn.input_dims[1]);
        clear_array(_layer->gradients_W, _layer->num_outputs, _layer->num_inputs);
        clear_array(ffn.errors, ffn.n_out_neurons, ffn.input_dims[1]);
        clear_array(_layer->gradients_B, _layer->num_outputs, 1);
    }

    if (verbose == 1) {

        printf("***********************************************************************\n");
        printf("weight are updated \n");
        printf("***********************************************************************\n\n");
    }
}

network fit_rnn(network ffn, double ***data_X, double ***target_Y, int num_iterations, int training_mode) {
    int i, j, r;
    int record_index;
    int minibach_index;
    double is_early_stop = 0;

    minibach_index = 0;
    for (i = 0; i < num_iterations && is_early_stop == 0; i++) {
        if (i % 50 == 0 || i == num_iterations - 1) {
            printf("processed: %f percent\n\n\n", ((double) (i + 1) / (double) num_iterations));
        }

        for (record_index = 0; record_index < ffn.num_records && is_early_stop == 0; record_index++) {
            forward_rnn(ffn, data_X[record_index]);
            backward_rnn(ffn, data_X[record_index], target_Y[record_index]);

            is_early_stop = 1;

            //print_network(ffn);
            break;
            //            //print_network(ffn);
            //            if (i % 100 == 0 && record_index == ffn.num_records - 1) {
            //                // check the correctness of gradient on the first iteration
            //                check_gradient(&ffn, data_X[record_index], target_Y[record_index]);
            //
            //                if (ffn.is_gradient_checked == 0) {
            //                    printf("Gradient is wrong. Break the training.\n");
            //                    is_early_stop = 1;
            //                    break;
            //                }
            //            }
            //
            //            if (verbose == 1) {
            //                printf("======================================================= iteration index %d \n", i);
            //                printf("======================================================= record index %d \n", record_index);
            //                printf("target output \n");
            //                print_vector(target_Y[record_index], ffn.n_out_neurons);
            //                printf("network output \n");
            //                print_matrix_double(ffn.layers[ffn.n_h_layers + 1].outputs, ffn.n_out_neurons, ffn.input_dims[1]);
            //            }
            //
            //
            //            if (minibach_index != 0 && (minibach_index % ffn.minibatch_size == 0)) {
            //                printf("*********************************errors ***********************\n\n");
            //                print_matrix_double(ffn.errors, ffn.n_out_neurons, ffn.input_dims[1]);
            //                printf("*********************************errors ***********************\n\n");
            //
            //                if (check_early_stopping(ffn) == 1) {
            //                    printf("======================================================= iteration index %d \n", i);
            //                    printf("EARLY STOPPING ACHIEVED \n");
            //                    is_early_stop = 1;
            //                    break;
            //                }
            //                update_weights(ffn, i);
            //                // reset minibatch index
            //                minibach_index = 0;
            //            }
            //            if (ffn.num_records > ffn.minibatch_size) {
            //                minibach_index++;
            //            }
        }

        // if number of records are smaller than batchsize than update weights after records iterations
        if (ffn.num_records < ffn.minibatch_size) {
            //            if (verbose == 0) {
            //                printf("*********************************errors ***********************\n");
            //                print_matrix_double(ffn.errors, ffn.n_out_neurons, ffn.input_dims[1]);
            //                printf("*********************************errors ***********************\n\n");
            //            }
            //
            //            if (check_early_stopping(ffn) == 1 && (i != num_iterations - 1)) {
            //                printf("======================================================= iteration index %d \n", i);
            //                printf("EARLY STOPPING ACHIEVED \n");
            //                is_early_stop = 1;
            //
            //                break;
            //            }
            //            update_weights(ffn, i);
        }
    }
}