/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.h to edit this template
 */

/* 
 * File:   math.h
 * Author: rodion
 *
 * Created on December 19, 2022, 9:48 AM
 */

#ifndef MATH_H
#define MATH_H

#ifdef __cplusplus
extern "C" {
#endif

    double** matrix_product(double **matrix1, double **matrix2, int row, int col, int col_k);
    double get_random_value();
    double** init_weight_matrix(int weight_r, int weight_c);
    double sigmoid_value(double value);
    double** sigmoid_to_matrix(double** matrix, int row, int col);
    double** matrix_subtract(double **matrix_1, double **matrix_2, int row, int col);
    double** matrix_sum(double **matrix_1, double **matrix_2, int row, int col);
    double** sigmoid_derivative(double **matrix, int row, int col);
    double** matrix_transpose(double **matrix, int row, int col);
    void print_matrix(double** matrix, int row, int col);
    double** hadamard_product(double ** matrix_1, double **matrix_2, int row, int col);
    double** multiply_scalar(double **matrix, double scalar, int row, int col);

#ifdef __cplusplus
}
#endif

#endif /* MATH_H */

