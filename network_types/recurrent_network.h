/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.h to edit this template
 */

/* 
 * File:   recurrent_network.h
 * Author: rodion
 *
 * Created on February 4, 2023, 11:23 AM
 */

#ifndef RECURRENT_NETWORK_H
#define RECURRENT_NETWORK_H

#ifdef __cplusplus
extern "C" {
#endif
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
            );

    struct layer* init_rnn_layers(
            int *input_dims,
            int num_input_params,
            int n_h_layers,
            int n_h_neurons,
            int n_,
            int activation,
            double bottleneck_value,
            int batch_size
            );

    network fit_rnn(network rnn, double ***data_X, double ***target_Y, int num_iterations, int training_mode);
    network* forward_rnn(network *rnn, double **data_X);
    network* forward_sequence(network *rrn, double **data_X, int d);

    network* backward_rnn(network *rrn, double **data_X, double **target_Y);
    network* backward_sequence(network *rnn, double **data_X, double **target_Y, int d);
    network* check_gradient_rnn(network *rrn, double **data_X, double **target_Y);
    int check_early_stopping_rnn(network *rrn);

    network* update_weights_rnn(network *rrn, int i_iteration);

    network* get_errors_rnn(network *rrn, double **output, double **target_Y, int n_row, int d);
    network* calculate_jacobi_matrix_rnn(network *rrn, layer *_layer, double **targetY, double **data_X, int d);

#ifdef __cplusplus
}
#endif

#endif /* RECURRENT_NETWORK_H */

