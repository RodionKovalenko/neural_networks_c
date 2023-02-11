/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.h to edit this template
 */

/* 
 * File:   feedforward_network.h
 * Author: rodion
 *
 * Created on February 4, 2023, 10:52 AM
 */

#ifndef FEEDFORWARD_NETWORK_H
#define FEEDFORWARD_NETWORK_H

#ifdef __cplusplus
extern "C" {
#endif

    network init_ffn(
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
   
#ifdef __cplusplus
}
#endif

#endif /* FEEDFORWARD_NETWORK_H */

