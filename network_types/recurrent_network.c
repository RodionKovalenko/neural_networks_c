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
#include "../network_types/network.h"
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
        hidden_layer->recurrent_weights = rnn_prev_weight_matrix;

        hidden_layer->outputs = build_array(hidden_layer->num_outputs, input_dims[1]);
        hidden_layer->layer_sequence_outputs = build_array_3d(batch_size, hidden_layer->num_outputs, input_dims[1]);
        hidden_layer->hidden_output_seq = build_array_3d(batch_size, hidden_layer->num_outputs, input_dims[1]);
        hidden_layer->bias = build_array(hidden_layer->num_outputs, 1);


        hidden_layer->gradients = build_array(hidden_layer->num_outputs, input_dims[1]);
        hidden_layer->gradients_recurrent = build_array(hidden_layer->num_outputs, input_dims[1]);
        hidden_layer->gradients_h_recurrent = build_array(hidden_layer->num_outputs, input_dims[1]);
        hidden_layer->gradients_B = build_array(hidden_layer->num_outputs, 1);
        hidden_layer->gradients_W = build_array(hidden_layer->num_outputs, hidden_layer->num_inputs);
        hidden_layer->gradients_W_recurrent = build_array(hidden_layer->num_outputs, hidden_layer->num_outputs);

        hidden_layer->prev_weights = build_array(hidden_layer->num_outputs, hidden_layer->num_inputs);
        hidden_layer->prev_gradients_W = build_array(hidden_layer->num_outputs, hidden_layer->num_inputs);
        hidden_layer->prev_recurrent_weights = build_array(hidden_layer->num_outputs, hidden_layer->num_outputs);
        hidden_layer->v_t = build_array(hidden_layer->num_outputs, hidden_layer->num_inputs);
        hidden_layer->q_t = build_array(hidden_layer->num_outputs, hidden_layer->num_inputs);
        hidden_layer->I = build_array(hidden_layer->num_outputs * hidden_layer->num_inputs, hidden_layer->num_outputs * hidden_layer->num_inputs);
        hidden_layer->I = init_identity_matrix(hidden_layer->I, hidden_layer->num_outputs * hidden_layer->num_inputs);
        hidden_layer->G_t = build_array(hidden_layer->num_outputs * hidden_layer->num_inputs, hidden_layer->num_outputs * hidden_layer->num_inputs);
        hidden_layer->G_t_1 = build_array(hidden_layer->num_outputs * hidden_layer->num_inputs, hidden_layer->num_outputs * hidden_layer->num_inputs);
        hidden_layer->G_t_2 = build_array(hidden_layer->num_outputs * hidden_layer->num_inputs, hidden_layer->num_outputs * hidden_layer->num_inputs);
        hidden_layer->hessian_1 = build_array(hidden_layer->num_outputs * hidden_layer->num_inputs, hidden_layer->num_outputs * hidden_layer->num_inputs);
        hidden_layer->hessian_2 = build_array(hidden_layer->num_outputs * hidden_layer->num_inputs, hidden_layer->num_outputs * hidden_layer->num_inputs);
        hidden_layer->gradient_W_Hessian = build_array(hidden_layer->num_outputs, hidden_layer->num_inputs);

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
    output_layer->layer_sequence_outputs = build_array_3d(batch_size, output_layer->num_outputs, input_dims[1]);
    output_layer->hidden_output_seq = build_array_3d(batch_size, output_layer->num_outputs, input_dims[1]);
    output_layer->bias = build_array(output_layer->num_outputs, 1);
    output_layer->gradients = build_array(output_layer->num_outputs, input_dims[1]);
    output_layer->gradients_recurrent = build_array(output_layer->num_outputs, input_dims[1]);
    output_layer->gradients_B = build_array(output_layer->num_outputs, 1);
    output_layer->gradients_W = build_array(output_layer->num_outputs, output_layer->num_inputs);

    output_layer->prev_weights = build_array(output_layer->num_outputs, output_layer->num_inputs);
    output_layer->prev_gradients_W = build_array(output_layer->num_outputs, output_layer->num_inputs);
    output_layer->prev_recurrent_weights = build_array(output_layer->num_outputs, output_layer->num_outputs);
    output_layer->v_t = build_array(output_layer->num_outputs, output_layer->num_inputs);
    output_layer->q_t = build_array(output_layer->num_outputs, output_layer->num_inputs);
    output_layer->I = build_array(output_layer->num_outputs * output_layer->num_inputs, output_layer->num_outputs * output_layer->num_inputs);
    output_layer->I = init_identity_matrix(output_layer->I, output_layer->num_outputs * output_layer->num_inputs);
    output_layer->G_t = build_array(output_layer->num_outputs * output_layer->num_inputs, output_layer->num_outputs * output_layer->num_inputs);
    output_layer->G_t_1 = build_array(output_layer->num_outputs * output_layer->num_inputs, output_layer->num_outputs * output_layer->num_inputs);
    output_layer->G_t_2 = build_array(output_layer->num_outputs * output_layer->num_inputs, output_layer->num_outputs * output_layer->num_inputs);
    output_layer->hessian_1 = build_array(output_layer->num_outputs * output_layer->num_inputs, output_layer->num_outputs * output_layer->num_inputs);
    output_layer->hessian_2 = build_array(output_layer->num_outputs * output_layer->num_inputs, output_layer->num_outputs * output_layer->num_inputs);
    output_layer->gradient_W_Hessian = build_array(output_layer->num_outputs, output_layer->num_inputs);

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

    network rnn = {
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

    printf("layer features %d \n", rnn.n_features);

    rnn.layers[0].outputs = build_array(1, rnn.n_features);

    if (verbose == 1) {
        print_network(rnn);
    }

    return rnn;
}

network* forward_rnn(network *rnn, double **data_X) {
    int d;

    for (d = 0; d < rnn->batch_size; d++) {
        rnn = forward_sequence(rnn, data_X, d);
    }

    return rnn;
}

network* forward_sequence(network *rnn, double **data_X, int d) {
    int l;
    double **layer_input, **outputs;
    layer *_layer, *_prev_layer;

    for (l = 1; l < rnn->n_h_layers + 2; l++) {
        _layer = &rnn->layers[l];
        _prev_layer = _layer->previous_layer;

        if (l == 1) {
            _prev_layer->outputs[0] = data_X[d];
        }

        if (_prev_layer != NULL) {
            if (d > 0 && _layer->layer_index > 2) {
                layer_input = _prev_layer->layer_sequence_outputs[d];
            } else {
                layer_input = _prev_layer->outputs;
            }

            outputs = _layer->layer_sequence_outputs[d];

            if (_prev_layer->layer_index == 1) {
                outputs = apply_matrix_product_transposed(outputs, _layer->weights, layer_input, _layer->num_outputs, rnn->input_dims[1], _layer->num_inputs);
            } else {
                outputs = apply_matrix_product(outputs, _layer->weights, layer_input, _layer->num_outputs, rnn->input_dims[1], _layer->num_inputs);
            }

            if (_layer->layer_sequence_outputs != NULL && _layer->recurrent_weights != NULL && d > 0 && (_layer->layer_index != rnn->n_h_layers + 2)) {
                _layer->hidden_output_seq[d] = apply_matrix_product(_layer->hidden_output_seq[d], _layer->recurrent_weights, _layer->layer_sequence_outputs[d - 1], _layer->num_outputs, rnn->input_dims[1], _layer->num_outputs);
                outputs = matrix_add_matrix(_layer->hidden_output_seq[d], outputs, _layer->num_outputs, rnn->input_dims[1]);
            }

            _layer->outputs = matrix_add_bias(outputs, _layer->bias, _layer->num_outputs, rnn->input_dims[1]);
            apply_activation(_layer, *rnn);

            _layer->layer_sequence_outputs[d] = copy_array(_layer->layer_sequence_outputs[d], _layer->outputs, _layer->num_outputs, rnn->input_dims[1]);
        }
    }

    return rnn;
}

// https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

network* backward_rnn(network *rnn, double **data_X, double **target_Y) {
    int d;

    for (d = rnn->batch_size - 1; d >= 0; d--) {
        rnn = backward_sequence(rnn, data_X, target_Y, d);
    }

    return rnn;
}

network* backward_sequence(network *rnn, double **data_X, double **target_Y, int d) {
    int l;
    layer *_layer;

    // calculate gradients
    for (l = rnn->n_h_layers + 1; l > 0; l--) {
        _layer = &rnn->layers[l];

        if (l == (rnn->n_h_layers + 1)) {
            rnn = get_errors_rnn(rnn, _layer->layer_sequence_outputs[d], target_Y, _layer->num_outputs, d);
        }

        rnn = calculate_jacobi_matrix_rnn(rnn, _layer, target_Y, data_X, d);
    }

    return rnn;
}

network* get_errors_rnn(network *ffn, double **output, double **target_Y, int n_row, int d) {
    int i, j;
    int normalizing_constant = ffn->minibatch_size;

    if (ffn->num_records < ffn->minibatch_size) {
        normalizing_constant = ffn->num_records;
    }

    for (i = 0; i < n_row; i++) {
        switch (ffn->loss_function) {
            case MEAN_SQUARED_ERROR_LOSS:
                ffn->errors[i][0] += pow((target_Y[d][i] - output[i][0]), 2.0) / (double) normalizing_constant;
                break;
        }
    }

    return ffn;
}

network* calculate_jacobi_matrix_rnn(network *ffn, layer *_layer, double **targetY, double **data_X, int s) {
    int i, j, k, n, p;
    int n_row = _layer->num_outputs;
    int d = ffn->input_dims[1];
    layer *_next_layer = _layer->next_layer;
    layer *_prev_layer = _layer->previous_layer;
    double derivative;


    for (i = 0; i < n_row; i++) {
        for (k = 0; k < d; k++) {
            derivative = apply_deactivation_to_value_rnn(_layer, i, k, s, *ffn);
            _layer->gradients[i][k] = 0;

            if (_layer->layer_index == (ffn->n_h_layers + 2)) {
                _layer->gradients[i][k] += -2.0 * (targetY[s][i] - _layer->layer_sequence_outputs[s][i][k]) * derivative;
            } else {
                for (n = 0; n < _next_layer->num_outputs; n++) {
                    _layer->gradients[i][k] += _next_layer->gradients[n][k] * _next_layer->weights[n][i] * derivative;
                }
            }

            _layer->gradients_B[i][0] += _layer->gradients[i][k];

            for (j = 0; j < _layer->num_inputs; j++) {
                if (_layer->layer_index == 2) {
                    _layer->gradients_W[i][j] += _layer->gradients[i][k] * data_X[s][j];
                } else {
                    _layer->gradients_W[i][j] += _layer->gradients[i][k] * _prev_layer->layer_sequence_outputs[s][j][k];
                }
            }

            if (_layer->gradients_W_recurrent != NULL && s > 0) {
                for (n = 0; n < _layer->num_outputs; n++) {
                    _layer->gradients_W_recurrent[i][n] += _layer->layer_sequence_outputs[s - 1][n][k] * _layer->gradients[i][k];
                }
            }
        }
    }

    return ffn;
}

network* update_weights_rnn(network *ffn, int i_iteration) {
    int i, j, r, l, k;
    int normalizing_constant = (double) ffn->minibatch_size;
    double p = 0.9999, pf = 0.9;
    double t = (double) i_iteration + 1;
    double prev_weight, prev_bias;
    double beta = 0.2;
    double lr = ffn->learning_rate;
    layer *_layer;
    int update_method = (int) rand() * 1.5;

    if (ffn->num_records < ffn->minibatch_size) {
        normalizing_constant = (double) ffn->num_records;
    }

    for (l = ffn->n_h_layers + 1; l >= 1; l--) {
        _layer = &ffn->layers[l];

        double normalizing_c;
        int n_row_flattened = _layer->num_outputs * _layer->num_inputs;
        if (_layer->prev_gradients_W[0][0] != 0.0 && _layer->prev_weights[0][0] != 0.0 || _layer->is_first_run_passed) {
            _layer->is_first_run_passed = 1;
            _layer->q_t = matrix_subtract(_layer->q_t, _layer->weights, _layer->prev_weights, _layer->num_outputs, _layer->num_inputs);
            _layer->v_t = matrix_subtract(_layer->v_t, _layer->gradients_W, _layer->prev_gradients_W, _layer->num_outputs, _layer->num_inputs);

            normalizing_c = get_normalizing_constant(_layer->q_t, _layer->v_t, _layer->num_outputs, _layer->num_inputs);

            //            _layer->hessian_1 = build_hessian(_layer->hessian_1, _layer->q_t, _layer->v_t, _layer->num_outputs, _layer->num_inputs);
            //            _layer->hessian_1 = multiply_scalar(_layer->hessian_1, normalizing_c, n_row_flattened, n_row_flattened);
            //            _layer->hessian_1 = matrix_subtract(_layer->hessian_1, _layer->I, _layer->hessian_1, n_row_flattened, n_row_flattened);
            //
            //            _layer->hessian_2 = build_hessian(_layer->hessian_2, _layer->v_t, _layer->q_t, _layer->num_outputs, _layer->num_inputs);
            //            _layer->hessian_2 = multiply_scalar(_layer->hessian_2, normalizing_c, n_row_flattened, n_row_flattened);
            //            _layer->hessian_2 = matrix_subtract(_layer->hessian_2, _layer->I, _layer->hessian_2, n_row_flattened, n_row_flattened);
            //
            //            _layer->G_t_1 = apply_matrix_product(_layer->G_t_1, _layer->hessian_1, _layer->G_t, n_row_flattened, n_row_flattened, n_row_flattened);
            //            _layer->G_t_2 = apply_matrix_product(_layer->G_t_2, _layer->G_t_1, _layer->hessian_2, n_row_flattened, n_row_flattened, n_row_flattened);

            _layer->G_t = build_hessian(_layer->G_t, _layer->q_t, _layer->q_t, _layer->num_outputs, _layer->num_inputs);
            _layer->G_t = multiply_scalar(_layer->G_t, normalizing_c, n_row_flattened, n_row_flattened);
            _layer->G_t = matrix_add_matrix(_layer->G_t, _layer->G_t_2, n_row_flattened, n_row_flattened);

            _layer->gradient_W_Hessian = multiply_with_hessian(_layer->gradient_W_Hessian, _layer->G_t, _layer->gradients_W, _layer->num_outputs, _layer->num_inputs);

            clear_array(_layer->G_t, n_row_flattened, n_row_flattened);
        } else {
            _layer->gradient_W_Hessian = copy_array(_layer->gradient_W_Hessian, _layer->gradients_W, _layer->num_outputs, _layer->num_inputs);
        }

        for (j = 0; j < _layer->num_outputs; j++) {

            if (_layer->layer_index >= 2 && _layer->recurrent_weights != NULL) {
                for (k = 0; k < _layer->num_outputs; k++) {
                    _layer->recurrent_weights[j][k] -= (lr * _layer->gradients_W_recurrent[j][k]) / (double) normalizing_constant;
                }
            }

            for (i = 0; i < _layer->num_inputs; i++) {
                prev_weight = _layer->weights[j][i];
                _layer->prev_gradients_W[j][i] = _layer->gradients_W[j][i];
                _layer->prev_weights[j][i] = _layer->weights[j][i];

                switch (ffn->optimizer) {
                    case DEFAULT:
                        // combination of Hessian optimization and gradient 
                        if (update_method > 1) {
                            _layer->weights[j][i] -= (lr * _layer->gradients_W[j][i]) / normalizing_constant;
                        } else {
                            _layer->weights[j][i] -= _layer->gradient_W_Hessian[j][i] / normalizing_constant;
                        }
                        break;
                    default:
                        _layer->weights[j][i] -= (lr * _layer->gradients_W[j][i]) / normalizing_constant;
                        break;
                }

                //_layer->weights[j][i] = beta * prev_weight + (1.0 - beta) * _layer->weights[j][i];
            }

            prev_bias = _layer->bias[j][0];

            switch (ffn->optimizer) {
                case DEFAULT:
                    _layer->bias[j][0] -= lr * _layer->gradients_B[j][0] / normalizing_constant;
                    break;
                default:
                    _layer->bias[j][0] -= lr * _layer->gradients_B[j][0] / normalizing_constant;
                    break;
            }

            // _layer->bias[j][0] = beta * prev_bias + (1.0 - beta) * _layer->bias[j][0];
        }

        clear_array(_layer->gradients, _layer->num_outputs, ffn->input_dims[1]);
        clear_array(_layer->gradients_recurrent, _layer->num_outputs, ffn->input_dims[1]);
        clear_array(_layer->gradients_W, _layer->num_outputs, _layer->num_inputs);
        clear_array(_layer->gradient_W_Hessian, _layer->num_outputs, _layer->num_inputs);

        clear_array(ffn->errors, ffn->n_out_neurons, ffn->input_dims[1]);
        clear_array(_layer->gradients_B, _layer->num_outputs, 1);

        if (_layer->gradients_W_recurrent != NULL) {
            clear_array(_layer->gradients_W_recurrent, _layer->num_outputs, _layer->num_outputs);
        }
    }

    if (verbose == 1) {
        printf("***********************************************************************\n");
        printf("weight are updated \n");
        printf("***********************************************************************\n\n");
    }

    return ffn;
}

network fit_rnn(network rnn, double ***data_X, double ***target_Y, int num_iterations, int training_mode) {
    int i, j, r;
    int record_index;
    int minibach_index;
    double is_early_stop = 0;

    minibach_index = 0;
    for (i = 0; i < num_iterations && is_early_stop == 0; i++) {
        if (i % 50 == 0 || i == num_iterations - 1) {
            printf("processed: %f percent\n\n\n", ((double) (i + 1) / (double) num_iterations));
        }

        for (record_index = 0; record_index < rnn.num_records && is_early_stop == 0; record_index++) {
            if (i % 100 == 0) {
                //check the correctness of gradient on the first iteration
                check_gradient_rnn(&rnn, data_X[record_index], target_Y[record_index]);

                if (rnn.is_gradient_checked == 0) {
                    printf("Gradient is wrong. Break the training.\n");
                    is_early_stop = 1;
                    break;
                }
            }

            rnn = *forward_rnn(&rnn, data_X[record_index]);
            rnn = *backward_rnn(&rnn, data_X[record_index], target_Y[record_index]);

            //  if (verbose == 1 || i == num_iterations - 1) {
            printf("======================================================= iteration index %d \n", i);
            printf("======================================================= record index %d \n", record_index);
            printf("target output \n");
            print_matrix_double(target_Y[record_index], rnn.batch_size, rnn.n_out_neurons);
            printf("network output \n");
            print_matrix_double_3d(rnn.layers[rnn.n_h_layers + 1].layer_sequence_outputs, rnn.batch_size, rnn.n_out_neurons, rnn.input_dims[1]);
            //  }

            if ((minibach_index != 0 && (minibach_index % rnn.minibatch_size == 0)) || rnn.minibatch_size == minibach_index + 1) {
                printf("*********************************errors ***********************\n\n");
                print_matrix_double(rnn.errors, rnn.n_out_neurons, rnn.input_dims[1]);
                printf("*********************************errors ***********************\n\n");

                if (check_early_stopping(rnn) == 1) {
                    printf("======================================================= iteration index %d \n", i);
                    printf("EARLY STOPPING ACHIEVED \n");
                    is_early_stop = 1;
                    break;
                }

                rnn = *update_weights_rnn(&rnn, i);
                // reset minibatch index
                minibach_index = 0;
            }

            if (rnn.num_records >= rnn.minibatch_size) {
                minibach_index++;
            }
        }

        // if number of records are smaller than batchsize than update weights after records iterations
        if (rnn.num_records < rnn.minibatch_size) {
            printf("*********************************errors ***********************\n\n");
            print_matrix_double(rnn.errors, rnn.n_out_neurons, rnn.input_dims[1]);
            printf("*********************************errors ***********************\n\n");

            if (check_early_stopping(rnn) == 1 && (i != num_iterations - 1)) {
                printf("======================================================= iteration index %d \n", i);
                printf("EARLY STOPPING ACHIEVED \n");
                is_early_stop = 1;

                break;
            }
            rnn = *update_weights_rnn(&rnn, i);
        }
    }
}

network* check_gradient_rnn(network *rnn, double **data_X, double **target_Y) {
    if (rnn->is_gradient_checked == 1) {
        return rnn;
    }
    // print_network(ffn);
    rnn->is_gradient_checked = 1;

    update_weights_rnn(rnn, 2);
    forward_rnn(rnn, data_X);
    backward_rnn(rnn, data_X, target_Y);

    int i, j, b;
    layer _first_h_layer = rnn->layers[1];
    layer _output_layer = rnn->layers[rnn->n_h_layers + 1];
    double ***original_outputs = build_array_3d(rnn->batch_size, rnn->n_out_neurons, rnn->input_dims[1]);

    double ***outputs = _output_layer.layer_sequence_outputs;
    double output_y1;
    double calculated_derivative_bias;
    double delta_x = 0.0000001;
    double tolerated_error = 0.001;

    double output_y2;
    double calculated_derivative;
    double derivative;

    for (b = 0; b < rnn->batch_size; b++) {
        for (j = 0; j < rnn->n_out_neurons; j++) {
            for (i = 0; i < rnn->input_dims[1]; i++) {
                original_outputs[b][j][i] = outputs[b][j][i];
            }
        }
    }

    printf("\n----------------------------------------- gradient check start--------------- \n\n\n");

    // test 1: output layer
    _output_layer.weights[0][0] += delta_x;
    forward_rnn(rnn, data_X);

    outputs = _output_layer.layer_sequence_outputs;
    derivative = 0;

    for (b = 0; b < rnn->batch_size; b++) {
        output_y1 = original_outputs[b][0][0];
        output_y2 = outputs[b][0][0];

        printf("orginal outputs : %f\n", output_y1);
        printf("similated outputs : %f\n", output_y2);
        printf("target outputs : %f\n", target_Y[b][0]);

        derivative += ((pow((target_Y[b][0] - output_y2), 2.0) - pow((target_Y[b][0] - output_y1), 2.0)) / delta_x);

        printf("simulated derivative : %f\n", derivative);
    }

    calculated_derivative = _output_layer.gradients_W[0][0];
    printf("approximated derivative of the first weight of output layer %f\n", derivative);
    printf("calculated derivative of first weight of output layer %f\n", calculated_derivative);

    if (fabs(fabs(derivative) - fabs(calculated_derivative)) < tolerated_error) {
        printf("\n\n\nderivative of the output layer is correct \n\n\n");
    } else {
        printf("GRADIENT IS NOT CORRECT\n");
        rnn->is_gradient_checked = 0;
    }

    _output_layer.weights[0][0] -= delta_x;

    // test 2: first hidden layer
    derivative = 0.0;

    for (b = 0; b < rnn->batch_size; b++) {
        _first_h_layer.weights[0][0] += delta_x;
        forward_sequence(rnn, data_X, b);
        outputs = _output_layer.layer_sequence_outputs;

        for (j = 0; j < _output_layer.num_outputs; j++) {
            for (i = 0; i < rnn->input_dims[1]; i++) {
                output_y2 = outputs[b][j][i];
                derivative += ((pow((target_Y[b][j] - output_y2), 2.0) - pow((target_Y[b][j] - original_outputs[b][j][i]), 2.0)) / delta_x);
            }
        }

        _first_h_layer.weights[0][0] -= delta_x;
        forward_sequence(rnn, data_X, b);
    }

    calculated_derivative = _first_h_layer.gradients_W[0][0];
    printf("approximated derivative of the first weight of first hidden layer %f\n", derivative);
    printf("calculated derivative of first weight of hidden layer %f\n", calculated_derivative);

    if (fabs(fabs(derivative) - fabs(calculated_derivative)) < tolerated_error) {
        printf("\n\n\nderivative of the hidden layer is correct \n\n\n");
    } else {
        printf("GRADIENT IS NOT CORRECT\n");
        rnn->is_gradient_checked = 0;
    }

    // test 3: last weight of hidden layer
    derivative = 0.0;

    for (b = 0; b < rnn->batch_size; b++) {
        _first_h_layer.weights[_first_h_layer.num_outputs - 1][0] += delta_x;
        forward_sequence(rnn, data_X, b);

        outputs = _output_layer.layer_sequence_outputs;
        for (j = 0; j < _output_layer.num_outputs; j++) {
            for (i = 0; i < rnn->input_dims[1]; i++) {
                output_y2 = outputs[b][j][i];
                derivative += ((pow((target_Y[b][j] - output_y2), 2.0) - pow((target_Y[b][j] - original_outputs[b][j][i]), 2.0)) / delta_x);
            }
        }

        _first_h_layer.weights[_first_h_layer.num_outputs - 1][0] -= delta_x;
        forward_sequence(rnn, data_X, b);
    }

    calculated_derivative = _first_h_layer.gradients_W[_first_h_layer.num_outputs - 1][0];
    printf("approximated derivative of the last weight of first hidden layer %f\n", derivative);
    printf("calculated derivative of last weight of hidden layer %f\n", calculated_derivative);

    if (fabs(fabs(derivative) - fabs(calculated_derivative)) < tolerated_error) {
        printf("\n\n\nderivative of the hidden layer is correct \n\n\n");
    } else {
        printf("GRADIENT IS NOT CORRECT\n");
        rnn->is_gradient_checked = 0;
    }

    // test 4: gradient of bias in output layer    
    _output_layer.bias[_output_layer.num_outputs - 1][0] += delta_x;
    forward_rnn(rnn, data_X);

    outputs = _output_layer.layer_sequence_outputs;
    derivative = 0;

    for (b = 0; b < rnn->batch_size; b++) {
        output_y2 = outputs[b][_output_layer.num_outputs - 1][0];
        output_y1 = original_outputs[b][_output_layer.num_outputs - 1][0];
        derivative += ((pow((target_Y[b][_output_layer.num_outputs - 1] - output_y2), 2.0) - pow((target_Y[b][_output_layer.num_outputs - 1] - output_y1), 2.0)) / delta_x);
    }

    calculated_derivative_bias = _output_layer.gradients_B[_output_layer.num_outputs - 1][0];
    printf("approximated bias derivative of the first weight of output layer %f\n", derivative);
    printf("bias calculated derivative of first weight of output layer %f\n", calculated_derivative_bias);

    if (fabs(fabs(derivative) - fabs(calculated_derivative_bias)) < tolerated_error) {
        printf("\n\n\nbias derivative of the output layer is correct \n\n\n");
    } else {
        printf("BIAS GRADIENT IS NOT CORRECT\n");
        rnn->is_gradient_checked = 0;
    }

    _output_layer.bias[_output_layer.num_outputs - 1][0] -= delta_x;

    // test 5: first weight of the recurrent hidden layer
    derivative = 0.0;

    for (b = 0; b < rnn->batch_size; b++) {
        _first_h_layer.recurrent_weights[0][0] += delta_x;
        forward_sequence(rnn, data_X, b);

        outputs = _output_layer.layer_sequence_outputs;
        for (j = 0; j < _output_layer.num_outputs; j++) {
            for (i = 0; i < rnn->input_dims[1]; i++) {
                output_y2 = outputs[b][j][i];
                derivative += ((pow((target_Y[b][j] - output_y2), 2.0) - pow((target_Y[b][j] - original_outputs[b][j][i]), 2.0)) / delta_x);
            }
        }

        _first_h_layer.recurrent_weights[0][0] -= delta_x;
        forward_sequence(rnn, data_X, b);
    }

    calculated_derivative = _first_h_layer.gradients_W_recurrent[0][0];
    printf("approximated derivative of the first weight of recurrent first hidden layer %f\n", derivative);
    printf("calculated derivative of the first weight of recurrent first hidden layer %f\n", calculated_derivative);

    if (fabs(fabs(derivative) - fabs(calculated_derivative)) < tolerated_error) {
        printf("\n\n\nderivative of the hidden layer is correct \n\n\n");
    } else {
        printf("GRADIENT IS NOT CORRECT\n");
        rnn->is_gradient_checked = 0;
    }

    // test 6: last weight of the recurrent hidden layer
    derivative = 0.0;

    int firstIndex = _first_h_layer.num_outputs - 1;
    for (b = 0; b < rnn->batch_size; b++) {
        _first_h_layer.recurrent_weights[firstIndex][0] += delta_x;
        forward_sequence(rnn, data_X, b);

        outputs = _output_layer.layer_sequence_outputs;
        for (j = 0; j < _output_layer.num_outputs; j++) {
            for (i = 0; i < rnn->input_dims[1]; i++) {
                output_y2 = outputs[b][j][i];
                derivative += ((pow((target_Y[b][j] - output_y2), 2.0) - pow((target_Y[b][j] - original_outputs[b][j][i]), 2.0)) / delta_x);
            }
        }

        _first_h_layer.recurrent_weights[firstIndex][0] -= delta_x;
        forward_sequence(rnn, data_X, b);
    }

    calculated_derivative = _first_h_layer.gradients_W_recurrent[firstIndex][0];
    printf("approximated derivative of the last weight of recurrent first hidden layer %f\n", derivative);
    printf("calculated derivative of the last weight of recurrent first hidden layer %f\n", calculated_derivative);

    if (fabs(fabs(derivative) - fabs(calculated_derivative)) < tolerated_error) {
        printf("\n\n\nderivative of the hidden layer is correct \n\n\n");
    } else {
        printf("GRADIENT IS NOT CORRECT\n");
        rnn->is_gradient_checked = 0;
    }
}