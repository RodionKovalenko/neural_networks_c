/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.h to edit this template
 */

/* 
 * File:   clear_memory.h
 * Author: rkovalenko
 *
 * Created on 12. Februar 2023, 16:56
 */

#ifndef CLEAR_MEMORY_H
#define CLEAR_MEMORY_H



#ifdef __cplusplus
extern "C" {
#endif

#include "../network_types/network.h"

    void clear_network(network ffn);
    void clear_matrix_memory(double **matrix, int rows);
    void clear_matrix_memory_3d(double ***matrix, int d1, int d2);

#ifdef __cplusplus
}
#endif

#endif /* CLEAR_MEMORY_H */

