/**
 * \file   loss.h
 * \author Domen Šoberl
 * \date   January 2023
 * \brief  Loss functions
 * 
 * This module provides a common interface for loss functions. It defines only
 * the MSE loss functions, but other functions can easily be defined if needed.
 * 
 * A loss function is of the format:
 * 
 *     double lossFunction(Matrix yhat, Matrix y, Matrix error)
 * 
 * - `yhat` (samples × output values):
 *     The predicted values (network output) for all samples in the batch.
 * - `y` (samples × true values):
 *     The true values for all samples in the batch.
 * - `error` (samples × error values):
 *     The error values for individual outputs and samples.
 *
 * Return value:
 *     The mean error over all outputs and samples.
 */

/**
 * When not using a loss function, provide the LOSS_NONE code. This defines
 * a function that treats values ´y´ as erros, i.e., copies the `y` values
 * directly to the ´error´ matrix.
 */
#define LOSS_NONE   0

/**
 * Mean Square Error function.
 */
#define LOSS_MSE    1

/**
 * A forward definition of the Matrix structure.
 */
typedef struct Matrix Matrix;

/**
 * Definition of the pointer to a loss function.
 */
typedef double (*LossFunction)(Matrix yhat, Matrix y, Matrix error);

/**
 * \returns The pointer to the loss function with the given `code`.
 */
LossFunction getLossFunction(int code);