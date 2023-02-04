/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.h to edit this template
 */

/* 
 * File:   network.h
 * Author: rodion
 *
 * Created on December 18, 2022, 6:40 PM
 */

#ifndef NETWORK_H
#define NETWORK_H

#ifdef __cplusplus
extern "C" {
#endif
    void set_verbose(int verbose);

    typedef struct network {
        //input array dimensions, i.e. number of [records][record dimension rows][record dimension columns]
        int *input_dims;
        int num_records;
        int activation_type;
        // number of input params, e.g 2 (rows and columns)
        int num_input_params;
        int gradient_check : 2;
        int is_gradient_checked;
        // number of hidden layers
        int n_h_layers;
        // number of hidden neurons in a hidden layer
        int n_h_neurons;
        // number of output neurons in the last output layer
        int n_out_neurons;
        // learning rate/
        double learning_rate;
        double learning_rate_b;
        double **errors;
        struct layer *layers;
        int minibatch_size;
        int loss_function;
        int optimizer;
        int networt_type;
    } network;

    network fit(network ffn, double **data_X, double **target_Y, int num_iterations, int training_mode);
    void forward(network ffn, double *data_X);
    void backward(network ffn, double *data_X, double *target_Y);
    void clear_network(network ffn);
    void clear_matrix_memory(double **matrix, int rows);
    void check_gradient(network *ffn, double *data_X, double *target_Y);
    int check_early_stopping(network ffn);

    void update_weights(network ffn, int i_iteration);

    double **get_errors(network ffn, double **output, double *target_Y, int n_row, int n_col);
    double **calculate_jacobi_matrix(network *ffn, layer *_layer, double **targetY, double **data_X);

    void update_weight_adam(network *ffn, layer *_layer, int normalizing_constant, double t, int i, int j, double p, double pf);
    void update_bias_adam(network *ffn, layer *_layer, int normalizing_constant, double t, int i, int j, double p, double pf);

#ifdef __cplusplus
}
#endif

#endif /* NETWORK_H */

