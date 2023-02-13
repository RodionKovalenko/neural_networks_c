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
            double bottleneck_value
            );

    struct layer* init_rnn_layers(
            int *input_dims,
            int num_input_params,
            int n_h_layers,
            int n_h_neurons,
            int n_,
            int activation,
            double bottleneck_value
            );
    
    network fit_rnn(network ffn, double ***data_X, double ***target_Y, int num_iterations, int training_mode);
    void forward_rnn(network ffn, double **data_X);
    void backward_rnn(network ffn, double **data_X, double **target_Y);
    void check_gradient_rnn(network *ffn, double **data_X, double **target_Y);
    int check_early_stopping_rnn(network ffn);

    void update_weights_rnn(network ffn, int i_iteration);

    double **get_errors_rnn(network ffn, double **output, double **target_Y, int n_row, int n_col);
    double **calculate_jacobi_matrix_rnn(network *ffn, layer *_layer, double **targetY, double **data_X);

#ifdef __cplusplus
}
#endif

#endif /* RECURRENT_NETWORK_H */

