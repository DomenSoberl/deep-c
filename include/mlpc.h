/**
 * \file   mlpc.h
 * \author Domen Šoberl
 * \date   January 2023
 * \brief  The public header that exposes the MLPC API.
 * 
 * This library implements Multilayer Perceptrons (MLPs) in pure C programming
 * language. It is suitable for the use in embedded systems or any AI related
 * project written in C/C++.
 * 
 * This header is intended to be used with the precompiled lib/mlpc static
 * library. Documentation for the exposed functions can be found within the
 * MLPC source code.
 */

#include <stdio.h>

#define ACTIVATION_NONE    0
#define ACTIVATION_LINEAR  0
#define ACTIVATION_SIGMOID 1
#define ACTIVATION_TANH    2
#define ACTIVATION_RELU    3

#define LOSS_NONE   0
#define LOSS_MSE    1

#define MATRIX(matrix, row, col) \
    matrix.data[row * matrix.columns + col]

typedef struct Matrix {
    int rows;
    int columns;
    double *data;
} Matrix;

Matrix matrix_create(int rows, int columns);
Matrix matrix_clone(Matrix matrix);
Matrix matrix_load(const char *filename);
int matrix_save(Matrix matrix, const char *filename);
void matrix_destroy(Matrix matrix);
void matrix_clear(Matrix matrix);
void matrix_copy(Matrix dst, Matrix src);
void matrix_fill(Matrix matrix, double value);
void matrix_randomize(Matrix matrix, double min, double max);

typedef struct MLP MLP;

void mlp_init();

MLP *mlp_create(
    int inputSize,
    int outputSize,
    int depth,
    int *hiddenLayerSizes,
    int hiddenLayerActivation,
    int outputLayerActivation,
    int batchSize);

MLP *mlp_clone(MLP *mlp);
void mlp_destroy(MLP *mlp);
void mlp_initialize(MLP *mlp);
void mlp_copy(MLP *src, MLP *dst);
Matrix mlp_feedforward(MLP *mlp, Matrix x);
double mlp_backpropagate(MLP *mlp, Matrix y, int lossFunctionId);
Matrix mlp_get_input_errors(MLP *mlp);
void mlp_sgd(MLP *mlp, double lr);
void mlp_sgd_clip(MLP *mlp, double lr, double clipnorm);
int mlp_load_weights(MLP *mlp, const char *filename);
int mlp_read_weights(MLP *mlp, FILE *file);
int mlp_save_weights(MLP *mlp, const char *filename);
int mlp_write_weights(MLP *mlp, FILE *file);

typedef struct Adam Adam;

Adam *adam_create(MLP *mlp);
void adam_destroy(Adam *adam);
void adam_set(Adam *adam, double alpha, double beta1, double beta2, double epsilon);
void adam_reset(Adam *adam);
void adam_optimize(MLP *mlp, Adam *adam);

int deepc_random_int(int min, int max);
double deepc_random_double(double min, double max);