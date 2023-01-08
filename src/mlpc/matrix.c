#include <malloc.h>
#include "matrix.h"
#include "random.h"

Matrix matrix_create(int rows, int columns)
{
    Matrix matrix;
    matrix.rows = rows;
    matrix.columns = columns;
    matrix.data = malloc(rows * columns * sizeof(double));
    
    return matrix;
}

Matrix matrix_clone(Matrix matrix)
{
    Matrix clone;
    clone.rows = matrix.rows;
    clone.columns = matrix.columns;
    clone.data = malloc(matrix.rows * matrix.columns * sizeof(double));

    for (int i = 0; i < matrix.rows * matrix.columns; i++)
        clone.data[i] = matrix.data[i];
    
    return clone;
}

Matrix matrix_load(const char *filename)
{
    Matrix matrix;
    matrix.rows = 0;
    matrix.columns = 0;
    matrix.data = NULL;

    FILE *file = fopen(filename, "rb");
    if (file == NULL)
        return matrix;
    
    matrix = matrix_read(file);
    fclose(file);

    return matrix;
}

Matrix matrix_read(FILE *file)
{
    Matrix matrix;
    matrix.rows = 0;
    matrix.columns = 0;
    matrix.data = NULL;

    int columns, rows, n;
    double *data;

    if (fread(&rows, sizeof(int), 1, file) != 1)
        return matrix;
    
    if (fread(&columns, sizeof(int), 1, file) != 1)
        return matrix;
    
    if ((n = rows * columns) <= 0)
        return matrix;
    
    if ((data = malloc(n * sizeof(double))) == NULL)
        return matrix;
    
    if (fread(data, sizeof(double), n, file) != n)
    {
        free(data);
        return matrix;
    }

    matrix.rows = rows;
    matrix.columns = columns;
    matrix.data = data;

    return matrix;
}

int matrix_save(Matrix matrix, const char *filename)
{
    FILE *file = fopen(filename, "wb");
    if (file == NULL)
        return -1;
    
    int result = matrix_write(matrix, file);
    fclose(file);
    
    return result;
}

int matrix_write(Matrix matrix, FILE *file)
{
    if (fwrite(&matrix.rows, sizeof(int), 1, file) != 1)
        return -1;
    
    if (fwrite(&matrix.columns, sizeof(int), 1, file) != 1)
        return -1;
    
    int n = matrix.rows * matrix.columns;
    if (fwrite(matrix.data, sizeof(double), n, file) != n)
        return -1;

    return 0;
}

void matrix_destroy(Matrix matrix)
{
    if (matrix.data != NULL)
        free(matrix.data);
}

void matrix_clear(Matrix matrix)
{
    for (int i = 0; i < matrix.rows * matrix.columns; i++)
        matrix.data[i] = 0;
}

void matrix_copy(Matrix dst, Matrix src)
{
    for (int i = 0; i < dst.rows * dst.columns; i++)
        dst.data[i] = src.data[i];
}

void matrix_fill(Matrix matrix, double value)
{
    for (int i = 0; i < matrix.rows * matrix.columns; i++)
        matrix.data[i] = value;
}

void matrix_randomize(Matrix matrix, double min, double max)
{
    for (int i = 0; i < matrix.rows * matrix.columns; i++)
        matrix.data[i] = deepc_random_double(min, max);
}

void matrix_sum(Matrix matrix1, Matrix matrix2, Matrix result)
{
    for (int i = 0; i < result.rows * result.columns; i++)
        result.data[i] = matrix1.data[i] + matrix2.data[i];
}

void matrix_add(Matrix dst, Matrix src)
{
    for (int i = 0; i < dst.rows * dst.columns; i++)
        dst.data[i] += src.data[i];
}

void matrix_difference(Matrix matrix1, Matrix matrix2, Matrix result)
{
    for (int i = 0; i < result.rows * result.columns; i++)
        result.data[i] = matrix1.data[i] - matrix2.data[i];
}

void matrix_subtract(Matrix dst, Matrix src)
{
    for (int i = 0; i < dst.rows * dst.columns; i++)
        dst.data[i] -= src.data[i];
}

void matrix_multiply(Matrix matrix, double value)
{
    for (int i = 0; i < matrix.rows * matrix.columns; i++)
        matrix.data[i] *= value;
}

void matrix_divide(Matrix matrix, double value)
{
    for (int i = 0; i < matrix.rows * matrix.columns; i++)
        matrix.data[i] /= value;
}

void matrix_odot(Matrix dst, Matrix src)
{
    for (int i = 0; i < dst.rows * dst.columns; i++)
        dst.data[i] *= src.data[i];
}

void matrix_dot(Matrix matrix1, Matrix matrix2, Matrix result)
{
    double *p = result.data;
    for (int row = 0; row < result.rows; row++)
    {
        for (int col = 0; col < result.columns; col++)
        {
            double *p1 = matrix1.data + matrix1.columns * row;
            double *p2 = matrix2.data + col;
            double sum = 0;
            for (int k = 0; k < matrix1.columns; k++)
            {
                sum += *p1 * *p2;
                p1 += 1;
                p2 += matrix2.columns;
            }
            *(p++) = sum;
        }
    }
}

void matrix_transpose(Matrix matrix, Matrix result)
{
     for (int row = 0; row < matrix.rows; row++)
        for (int col = 0; col < matrix.columns; col++)
            result.data[col * result.columns + row] = matrix.data[row * matrix.columns + col];
}

void matrix_dot_transpose(Matrix matrix1, Matrix matrix2, Matrix result)
{
    for (int col = 0; col < result.columns; col++)
    {
        for (int row = 0; row < result.rows; row++)
        {
            double *p1 = matrix1.data + matrix1.columns * col;
            double *p2 = matrix2.data + row;
            double sum = 0;
            for (int k = 0; k < matrix1.columns; k++)
            {
                sum += *p1 * *p2;
                p1 += 1;
                p2 += matrix2.columns;
            }
            result.data[row * result.columns + col] = sum;
        }
    }
}

void matrix_sum_rows_transpose(Matrix matrix, Matrix result)
{
    for (int col = 0; col < matrix.columns; col++)
    {
        int idx = col * result.columns;
        result.data[idx] = 0;
        for (int row = 0; row < matrix.rows; row++)
            result.data[idx] += matrix.data[row * matrix.columns + col];
    }
    for (int col = 1; col < result.columns; col++)
    {
        for (int row = 0; row < result.rows; row++)
            result.data[row * result.columns + col] = result.data[row * result.columns];
    }
}

void matrix_apply(Matrix matrix, ActivationFunction activationFunction)
{
    for (int i = 0; i < matrix.rows * matrix.columns; i++)
        matrix.data[i] = activationFunction(matrix.data[i]);
}