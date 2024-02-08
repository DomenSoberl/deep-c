/**
 * \file   adam.h
 * \author Domen Šoberl
 * \date   February 2024
 * \brief  Adam optimizer
 * 
 * This unit implements the Adam optimizer. As opposed to the SGD optimization,
 * which is implemented within the MLP unit, Adam optimization requires its
 * internal state to be stored and transferred between consecutive optimization
 * steps.
 */

#include <stdio.h>
#include "mlp.h"

/**
 * The Adam structure that is bound to a specific MLP and used only with it.
 */
typedef struct Adam
{
    /**
     * Step counter, that increases each time the MLP is being optimized.
     */
    double t;

    /**
     * The learning rate parameter alpha. Default value: 0.001.
     */
    double alpha;

    /**
     * The decay rate parameter beta1. default value: 0.9.
     */
    double beta1;

    /**
     * The decay rate parameter beta2. default value: 0.999.
     */
    double beta2;

    /**
     * Decay rate beta1 at step t.
     */
    double beta1t;

    /**
     * Decay rate beta2 at step t.
     */
    double beta2t;

    /**
     * The epsilon parameter. Default value: 1e-7.
     */
    double epsilon;

    /**
     * The number of layers within the MLP, including the output layer. In
     * comparison to the MLP depth value, this values is greater by 1.
     */
    int depth;

    /**
     * The moment vector for weights at time t. An array of matrices, one
     * matrix for each layer.
     */
    Matrix *mw;
    
    /**
     * The moment vector for biases at time t. An array of matrices, one
     * matrix for each layer.
     */
    Matrix *mb;

    /**
     * The infinity norm for weights at time t. An array of matrices, one
     * matrix for each layer.
     */
    Matrix *vw;

    /**
     * The infinity norm for biases at time t. An array of matrices, one
     * matrix for each layer.
     */
    Matrix *vb;
} Adam;

/**
 * Creates a new Adam structure, compatible with the given MLP structure. Every
 * Adam structure created with this function must eventually be destroyed by
 * calling `adam_destroy`.
 * 
 * \returns The newly created and initialized Adam structure.
 */
Adam *adam_create(MLP *mlp);

/**
 * Frees the memory allocated by the given Adam structure. After the Adam
 * structure has been destroyed, it must not be used anymore.
 */
void adam_destroy(Adam *adam);

/**
 * Enables the user to change the arbitrarily set the Adam parameters: alpha,
 * beta1, beta2, and epsilon. This function should be called immediately upon
 * creation, before the structure is used for the first time.
 */
void adam_set(Adam *adam, double alpha, double beta1, double beta2, double epsilon);

/**
 * Resets the time steps to 0. This means reseting all the time-sensitive
 * variables back to their initial values. If the default values of the Adam
 * parameters have been changed through the `adam_set` function, the user
 * defined values are taken into account.
 */
void adam_reset(Adam *adam);

/**
 * Performs one optimization step on the given `mlp` with the given `adam`
 * optimizer.
 */
void adam_optimize(MLP *mlp, Adam *adam);