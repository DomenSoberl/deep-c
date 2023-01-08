/**
 * \file   saddle.c
 * \author Domen Šoberl
 * \date   January 2023
 * \brief  Training a saddle function.
 * 
 * This is a simple example of how to use the MLPC library. We are using
 * a neural network with one hidden layer of 64 neurons to learn the saddle
 * function y = x1^2 - x2^2.
 */

#include <stdio.h>
#include "mlpc.h"

/**
 * A definition of the saddle function.
 */
double f(double x1, double x2)
{
    return x1*x1 - x2*x2;
}

/**
 * A function that generates a batch of random points within the [-1, 1] domain
 * of the saddle function.
 * 
 * \param x A matrix to be populated with randomly selected values (x1, x2).
 * \param y A matrix to be populated with values y = f(x1, x2).
 */
void sample(Matrix x, Matrix y)
{
    /* Fill the x matrix with random values. */
    matrix_randomize(x, -1, 1);
    
    /* Set the y = f(x1, x2) values. */
    for (int row = 0; row < y.rows; row++)
        MATRIX(y, row, 0) = f(MATRIX(x, row, 0), MATRIX(x, row, 1));
}

int main()
{
    /* Initialize the MLPC library. */
    mlp_init();

    /* We define a neural network with the following properties:
       - Two inputs, one output.
       - One hidden layer with 64 neurons.
       - ReLu activation function for the hidden layer.
       - Linear activation function for the output layer.
       - Batch size is 32 samples.
    */
    int layerSizes[] = {64};
    MLP *mlp = mlp_create(2, 1, 1, layerSizes, ACTIVATION_RELU, ACTIVATION_LINEAR, 32);
    
    /* The input matrix (rows = 32 samples, columns = {x1, x2}). */
    Matrix x = matrix_create(32, 2);
    
    /* The output matrix (rows = 32 samples, columns = {y}). */
    Matrix y = matrix_create(32, 1);

    /* Adam optimizer using the default values. */
    Adam *adam = adam_create(mlp);

    /* Train for 10000 steps. */
    double loss = 0;
    for (int i = 1; i <= 10000; i++)
    {
        /* Create a random batch of 32 samples. */
        sample(x, y);    

        /* Feed forward the generated x values. Here we ignore the returned predictions. */
        mlp_feedforward(mlp, x);
        
        /* Back-propagate using the MSE loss function on the true values y. */
        loss += mlp_backpropagate(mlp, y, LOSS_MSE);
        
        /* Optimize the neural network with Adam. */
        adam_optimize(mlp, adam);

        /* Alternatively, use SGD instead of Adam. */
        // mlp_sgd(mlp, 0.01);

        /* Print the mean loss over the last 100 steps. */
        if (i % 100 == 0)
        {
            printf("%d %f\n", i, loss / 100);
            loss = 0;
        }
    }

    /* Destroy all the data structures that we created explicitly. */
    mlp_destroy(mlp);
    matrix_destroy(x);
    matrix_destroy(y);
    adam_destroy(adam);

    return 0;
}