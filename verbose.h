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
#include "layer.h"
#include "feedforward_network.h"

    void print_layer(struct layer _layer);
    void print_network(feedforward_network network);

#ifdef __cplusplus
}
#endif

#endif /* VERBOSE_H */

