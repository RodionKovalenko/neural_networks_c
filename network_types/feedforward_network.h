/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.h to edit this template
 */

/* 
 * File:   feedforward_network.h
 * Author: rodion
 *
 * Created on December 18, 2022, 6:40 PM
 */

#ifndef FEEDFORWARD_NETWORK_H
#define FEEDFORWARD_NETWORK_H

#ifdef __cplusplus
extern "C" {
#endif
    void set_verbose(int verbose);

    typedef struct feedforward_network {
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
    } feedforward_network;

    feedforward_network init_ffn(
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

    struct layer* init_layers(
            int *input_dims,
            int num_input_params,
            int n_h_layers,
            int n_h_neurons,
            int n_,
            int activation,
            double bottleneck_value
            );

    feedforward_network fit(feedforward_network ffn, double **data_X, double **target_Y, int num_iterations, int training_mode);
    void forward(feedforward_network ffn, double *data_X);
    double **apply_activation(layer *_layer, feedforward_network ffn);
    double **apply_deactivation(layer *_layer, feedforward_network ffn);
    double apply_deactivation_to_value(layer *_layer, int row, int column, feedforward_network ffn);
    void update_weights(feedforward_network ffn, int i_iteration);
    void clear_network(feedforward_network ffn);
    void clear_matrix_memory(double **matrix, int rows);
    void check_gradient(feedforward_network *ffn, double *data_X, double *target_Y);
    int check_early_stopping(feedforward_network ffn);

    double **get_errors(feedforward_network ffn, double **output, double *target_Y, int n_row, int n_col);
    double **calculate_jacobi_matrix(feedforward_network *ffn, layer *_layer, double **targetY, double **data_X);
    
    void update_weight_adam(feedforward_network *ffn, layer *_layer, int normalizing_constant, double t, int i, int j, double p, double pf);
    void update_bias_adam(feedforward_network *ffn, layer *_layer, int normalizing_constant, double t, int i, int j, double p, double pf);

#ifdef __cplusplus
}
#endif

#endif /* FEEDFORWARD_NETWORK_H */

