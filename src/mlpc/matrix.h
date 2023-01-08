/**
 * \file   matrix.h
 * \author Domen Šoberl
 * \date   January 2023
 * \brief  Matrix structure and arithmetics
 *
 * This module provides a low-level implementation of 2D matrices. The matrices
 * are of fixed shape, which is determined upon matrix creation. Entries are
 * always of type double. Arithmetic operations are performed without checking
 * the validity of dimensions and it is up to the user to provide matrices of
 * valid shapes for the used operation. Memory violation may occur otherwise.
 * 
 * A matrix is created by calling the matrix_create function and must finally be
 * disposed by calling the matrix_destroy function.
 * 
 * The elements are stored in a continuous block of memory as an array of type
 * double. The element at (`row`, `col`) position can be be found at the
 * `[row * matrix.columns + col]` index in the array.
 */

#include <stdio.h>
#include "activation.h"

/**
 * A macro for an easier access to the (`row`, `col`) element.
 */
#define MATRIX(matrix, row, col) \
    matrix.data[row * matrix.columns + col]

/**
 * The matrix structure contains a pointer to the heap-allocated array of type
 * double. The number of rows and columns are stored as integers. This structure
 * would typically be passed by value, which means the the shape information is
 * copied through stack, while the contents of the matrix remains on the heap.
 */
typedef struct Matrix {
    /**
     * The number of rows - the height of the matrix.
     */
    int rows;

    /**
     * The number of columns - the width of the matrix.
     */
    int columns;

    /**
     * The pointer to the data - an array that contains rows * columns values
     * of type double.
     */
    double *data;
} Matrix;

/**
 * Creates a matrix with the given shape. Every matrix created with this
 * function must eventually be destroyed by calling `matrix_destroy`.
 * 
 * \returns The newly created Matrix.
 */
Matrix matrix_create(int rows, int columns);

/**
 * Creates a clone of the given matrix by copying all its contents to a newly
 * created matrix of the same shape. Every matrix created with this function
 * must eventually be destroyed by calling `matrix_destroy`.
 * 
 * \returns The newly created Matrix.
 */
Matrix matrix_clone(Matrix matrix);

/**
 * Loads a matrix from a file. If the file cannot be read, an empty matrix is
 * created an returned. The returned matrix must eventually be destroyed by
 * calling `matrix_destroy`.
 * 
 * \returns The newly created Matrix.
 */
Matrix matrix_load(const char *filename);

/**
 * Loads a matrix from an opened file stream. If the file stream cannot be read,
 * an empty matrix is created an returned. The returned matrix must eventually
 * be destroyed by calling `matrix_destroy`.
 * 
 * \returns The newly created Matrix.
 */
Matrix matrix_read(FILE *file);

/**
 * Stores the content of the matrix to a binary file.
 * 
 * \returns 0 if no errors, -1 if errors.
 */
int matrix_save(Matrix matrix, const char *filename);

/**
 * Writes the content of the matrix to an opened binary stream.
 * 
 * \returns 0 if no errors, -1 if errors.
 */
int matrix_write(Matrix matrix, FILE *file);

/**
 * Frees the heap memory allocated by the given matrix. After the matrix is
 * destroyed, it must not be used anymore.
 */
void matrix_destroy(Matrix matrix);

/**
 * Sets all elements in the given `matrix` to zero.
 */
void matrix_clear(Matrix matrix);

/**
 * Copies the contents of the `src` matrix to the `dst` matrix. The shape of
 * both matrices must match.
 */
void matrix_copy(Matrix dst, Matrix src);

/**
 * Sets all elements in the given `matrix` to the given `value`.
 */
void matrix_fill(Matrix matrix, double value);

/**
 * Fills the given `matrix` with random values between `min` and `max`.
 */
void matrix_randomize(Matrix matrix, double min, double max);

/**
 * Sums matrices `matrix1` and `matrix2´ element-wise and stores the result to
 * the `result` matrix. All three matrices must be of the same shape.
 */
void matrix_sum(Matrix matrix1, Matrix matrix2, Matrix result);

/**
 * Adds the `src` matrix to the `dst` matrix element-wise. Both matrices must
 * be of the same shape.
 */
void matrix_add(Matrix dst, Matrix src);

/**
 * Subtracts `matrix2` from `matrix1` element-wise and stores the result to the
 * `result` matrix. All three matrices must be of the same shape.
 */
void matrix_difference(Matrix matrix1, Matrix matrix2, Matrix result);

/**
 * Subtracts matrix `src` from matrix `dst`. Both matrices must be of the
 * same shape.
 */
void matrix_subtract(Matrix dst, Matrix src);

/**
 * Multiplies all elements in the given `matrix` with the given `value`.
 */
void matrix_multiply(Matrix matrix, double value);

/**
 * Divides all elements in the given `matrix` with the given `value`.
 */
void matrix_divide(Matrix matrix, double value);

/**
 * Multiplies the elements in the `dst` matrix with the elements at the
 * corresponding positions in the `src` matrix. Both matrices must be
 * of the same shape.
 */
void matrix_odot(Matrix dst, Matrix src);

/**
 * Multiplies matrices `matrix1` and `matrix2' and stores the result to
 * the `result` matrix. The shape of the given matrices must be in
 * compliance with the matrix multiplication rule, i.e., the number
 * of columns of `matrix1` must be equal to the number of rows of
 * `matrix2`. The `result` matrix must have the same number of rows
 * as `matrix1` and the same number of columns as `matrix2`.
 */
void matrix_dot(Matrix matrix1, Matrix matrix2, Matrix result);

/**
 * Transposes the `matrix` and stores the transposed content to the
 * `result` matrix.
 */
void matrix_transpose(Matrix matrix, Matrix result);

/**
 * Multiplies matrices `matrix1` and `matrix2` and transposes the result.
 * It is a composite of functions `matrix_dot` and `matrix_transpose`.
 * The shapes of the matrices must obey the same rules as with the
 * `matrix_dot` function, except that the dimensions of the `result`
 * matrix are inverted.
 */
void matrix_dot_transpose(Matrix matrix1, Matrix matrix2, Matrix result);

/**
 * This function is a composite of 3 operations:
 *   1. Sum all the elements in the same column. This way a single row
 *      matrix is obtained.
 *   2. Repeat the resulting row so that the original matrix height is
 *      obtained.
 *   3. Transpose the result.
 * 
 * This function is defined for easier computation of gradients during
 * back-propagation in neural networks.
 */
void matrix_sum_rows_transpose(Matrix matrix, Matrix result);

/**
 * Applies the given activation function to every element in the `matrix`.
 */
void matrix_apply(Matrix matrix, ActivationFunction activationFunction);