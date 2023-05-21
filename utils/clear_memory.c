/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.c to edit this template
 */
#include <stdlib.h>
#include <stdio.h>
#include "../network_types/network.h"
#include "../network_types/layer.h"
#include "clear_memory.h"
#include "verbose.h"

void clear_matrix_memory_3d(double ***matrix, int d1, int d2) {
    int i, j;

    for (i = 0; i < d1; i++) {
        clear_matrix_memory(matrix[i], d2);
    }

    free(matrix);
}

void clear_matrix_memory(double **matrix, int rows) {
    int i;

    for (i = 0; i < rows; i++) {
        free(matrix[i]);
    }

    free(matrix);
}

void clear_network(network ffn) {
    int l = 0;
    layer *_layer;



    for (l = ffn.n_h_layers + 1; l >= 1; l--) {
        _layer = &ffn.layers[l];
        if (ffn.layers[l].errors != NULL) {
            clear_matrix_memory(ffn.layers[l].errors, ffn.n_out_neurons);
        }

        free(_layer->next_layer);

        clear_matrix_memory(_layer->gradients, _layer->num_outputs);
        clear_matrix_memory(_layer->gradients_W, _layer->num_outputs);
        clear_matrix_memory(_layer->bias, _layer->num_outputs);
        clear_matrix_memory(_layer->weights, _layer->num_outputs);

        if (_layer->outputs != NULL) {
            clear_matrix_memory(_layer->outputs, _layer->num_outputs);
        }

        if (_layer->prev_weights != NULL) {
            clear_matrix_memory(_layer->prev_weights, _layer->num_outputs);
        }

        if (_layer->recurrent_weights != NULL) {
            clear_matrix_memory(_layer->recurrent_weights, _layer->num_outputs);
        }
        if (_layer->prev_recurrent_weights != NULL) {
            clear_matrix_memory(_layer->prev_recurrent_weights, _layer->num_outputs);
        }
        if (_layer->gradients_h_recurrent != NULL) {
            clear_matrix_memory(_layer->gradients_h_recurrent, _layer->num_outputs);
        }
        if (_layer->gradients_recurrent != NULL) {
            clear_matrix_memory(_layer->gradients_recurrent, _layer->num_outputs);
        }
        if (_layer->gradients_B != NULL) {
            clear_matrix_memory(_layer->gradients_B, _layer->num_outputs);
        }
        if (_layer->adam_A_bias != NULL) {
            clear_matrix_memory(_layer->adam_A_bias, _layer->num_outputs);
        }

        if (_layer->adam_B_bias != NULL) {
            clear_matrix_memory(_layer->adam_B_bias, _layer->num_outputs);
        }
        if (_layer->prev_gradients_W != NULL) {
            clear_matrix_memory(_layer->prev_gradients_W, _layer->num_outputs);
        }
        if (_layer->gradients_W_recurrent != NULL) {
            clear_matrix_memory(_layer->gradients_W_recurrent, _layer->num_outputs);
        }
        if (_layer->prev_gradient_W_reccurent != NULL) {
            clear_matrix_memory(_layer->prev_gradient_W_reccurent, _layer->num_outputs);
        }
        if (_layer->q_t != NULL) {
            clear_matrix_memory(_layer->q_t, _layer->num_outputs);
        }
        if (_layer->v_t != NULL) {
            clear_matrix_memory(_layer->v_t, _layer->num_outputs);
        }
        if (_layer->I != NULL) {
            clear_matrix_memory(_layer->I, _layer->num_outputs * _layer->num_inputs);
        }
        if (_layer->G_t_1 != NULL) {
            clear_matrix_memory(_layer->G_t_1, _layer->num_outputs * _layer->num_inputs);
        }

        if (_layer->G_t_2 != NULL) {
            clear_matrix_memory(_layer->G_t_2, _layer->num_outputs * _layer->num_inputs);
        }
        if (_layer->G_t != NULL) {
            clear_matrix_memory(_layer->G_t, _layer->num_outputs * _layer->num_inputs);
        }
        if (_layer->gradient_B_Hessian != NULL) {
            clear_matrix_memory(_layer->gradient_B_Hessian, _layer->num_outputs);
        }
        if (_layer->gradient_W_Hessian != NULL) {
            clear_matrix_memory(_layer->gradient_W_Hessian, _layer->num_outputs);
        }
        if (_layer->gradient_W_recurrent_Hessian != NULL) {
            clear_matrix_memory(_layer->gradient_W_recurrent_Hessian, _layer->num_outputs);
        }
        if (_layer->hessian_1 != NULL) {
            clear_matrix_memory(_layer->hessian_1, _layer->num_outputs * _layer->num_inputs);
        }
        if (_layer->hessian_2 != NULL) {
            clear_matrix_memory(_layer->hessian_2, _layer->num_outputs * _layer->num_inputs);
        }
        if (_layer->adam_A != NULL) {
            clear_matrix_memory(_layer->adam_A, _layer->num_outputs);
        }
        if (_layer->adam_B != NULL) {
            clear_matrix_memory(_layer->adam_B, _layer->num_outputs);
        }

        if (_layer->hidden_output_seq != NULL) {
            clear_matrix_memory_3d(_layer->hidden_output_seq, ffn.batch_size, _layer->num_outputs);
        }
        if (_layer->layer_sequence_outputs != NULL) {
            clear_matrix_memory_3d(_layer->layer_sequence_outputs, ffn.batch_size, _layer->num_outputs);
        }
    }

    free(ffn.layers);
}

