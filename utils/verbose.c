/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.c to edit this template
 */

#include <stdio.h>
#include <stdlib.h>
#include "../layer.h"
#include "../feedforward_network.h"

void print_layer(layer _layer) {
    printf("test");
}

void print_network(feedforward_network network) {
    printf("verbose mode \n\n");
    int i, j;

    layer *layers = network.layers;
    int n_layers = network.n_h_layers;

    for (i = 0; i < n_layers + 2; i++) {
        printf("layer %d \n", i + 1);

        layer *_layer = (layers + i);

        if (_layer == NULL) {
            printf("layer is null");
            continue;
        }

        printf("weight matrix size: (%d x %d) \n", _layer->num_outputs, _layer->num_inputs);
        
        printf("number of inputs: (%d => %d) \n", _layer->num_inputs, _layer->num_outputs);
    }

}