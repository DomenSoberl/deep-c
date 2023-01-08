#include <math.h>
#include "loss.h"
#include "matrix.h"

double errorFunction(Matrix yhat, Matrix y, Matrix error)
{
    matrix_copy(error, y);

    double sum = 0;
    int n = error.rows * error.columns;
    for (int i = 0; i < n; i++)
        sum += error.data[i];

    return sum / n;
}

double mse(Matrix yhat, Matrix y, Matrix error)
{
    matrix_difference(yhat, y, error);

    double sum = 0;
    int n = error.rows * error.columns;
    for (int i = 0; i < n; i++)
        sum += pow(error.data[i], 2);
    
    return sum / (double)n;
}

LossFunction getLossFunction(int code)
{
    switch (code)
    {
        case LOSS_MSE:
            return mse;
        default:
            return errorFunction;
    }
}