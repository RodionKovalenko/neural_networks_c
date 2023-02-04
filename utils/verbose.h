/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.h to edit this template
 */

/* 
 * File:   verbose.h
 * Author: rodion
 *
 * Created on December 19, 2022, 9:53 AM
 */


#ifndef VERBOSE_H
#define VERBOSE_H

#ifdef __cplusplus
extern "C" {
#endif
#include "../network_types/layer.h"
#include "../network_types/network.h"

    void print_layer(layer *_layer);
    void print_network(network network);
    void print_matrix_int(int **matrix, int rows, int columns);
    void print_matrix_double(double **matrix, int rows, int columns);
    void print_forward_updates(network ffn, layer *_layer);
    void print_vector(double *matrix, int columns);

#ifdef __cplusplus
}
#endif

#endif /* VERBOSE_H */

