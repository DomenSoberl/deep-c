#include <malloc.h>
#include <math.h>
#include "random.h"
#include "mlp.h"
#include "loss.h"

/* Initialize the MLPC library. */
void mlp_init()
{
    deepc_random_init();
}

/* A helper function to create a layer. */
Layer mlp_create_layer(int inputSize, int outputSize, int batchSize, int activation)
{
    Layer layer;
    layer.weights = matrix_create(outputSize, inputSize);
    layer.biases = matrix_create(outputSize, batchSize);
    layer.output = matrix_create(outputSize, batchSize);
    layer.errors = matrix_create(batchSize, outputSize);
    layer.deltas = matrix_create(batchSize, outputSize);
    layer.gradWeights = matrix_create(outputSize, inputSize);
    layer.gradBiases = matrix_create(outputSize, batchSize);
    layer.activation = getActivationFunction(activation);
    layer.activationDeriv = getActivationFunctionDeriv(activation);
    
    return layer;
}

/* Creates a new neural network on heap. */
MLP *mlp_create(int inputSize, int outputSize, int depth, int *hiddenLayerSizes, int hiddenLayerActivation, int outputLayerActivation, int batchSize)
{
    MLP *mlp = malloc(sizeof(MLP));
    mlp->depth = depth;
    mlp->batchSize = batchSize;
    mlp->layers = malloc((depth + 1) * sizeof(Layer));
    
    int layerInputSize = inputSize;
    for (int i = 0; i < depth; i++)
    {
        mlp->layers[i] = mlp_create_layer(layerInputSize, hiddenLayerSizes[i], batchSize, hiddenLayerActivation);
        layerInputSize = hiddenLayerSizes[i];
    }
    mlp->layers[depth] = mlp_create_layer(layerInputSize, outputSize, batchSize, outputLayerActivation);

    mlp->input = matrix_create(inputSize, batchSize);
    mlp->inputErrors = matrix_create(batchSize, inputSize);
    mlp->output = matrix_create(batchSize, outputSize);

    mlp_initialize(mlp);
    
    return mlp;
}

/* Creates a new neural network on heap that is a clone of the given neural network. */
MLP *mlp_clone(MLP *mlp)
{
    MLP *clone = malloc(sizeof(MLP));
    clone->depth = mlp->depth;
    clone->batchSize = mlp->batchSize;
    clone->layers = malloc((mlp->depth + 1) * sizeof(Layer));

    for (int i = 0; i <= mlp->depth; i++)
    {
        clone->layers[i].weights = matrix_clone(mlp->layers[i].weights);
        clone->layers[i].biases = matrix_clone(mlp->layers[i].biases);
        clone->layers[i].output = matrix_clone(mlp->layers[i].output);
        clone->layers[i].errors = matrix_clone(mlp->layers[i].errors);
        clone->layers[i].deltas = matrix_clone(mlp->layers[i].deltas);
        clone->layers[i].gradWeights = matrix_clone(mlp->layers[i].gradWeights);
        clone->layers[i].gradBiases = matrix_clone(mlp->layers[i].gradBiases);
        clone->layers[i].activation = mlp->layers[i].activation;
        clone->layers[i].activationDeriv = mlp->layers[i].activationDeriv;
    }

    clone->input = matrix_clone(mlp->input);
    clone->inputErrors = matrix_clone(mlp->inputErrors);
    clone->output = matrix_clone(mlp->output);

    return clone;
}

/* Destroys a neural network created with mlp_create or mlp_clone. */
void mlp_destroy(MLP *mlp)
{
    for (int i = 0; i <= mlp->depth; i++)
    {
        matrix_destroy(mlp->layers[i].weights);
        matrix_destroy(mlp->layers[i].biases);
        matrix_destroy(mlp->layers[i].output);
        matrix_destroy(mlp->layers[i].errors);
        matrix_destroy(mlp->layers[i].deltas);
        matrix_destroy(mlp->layers[i].gradWeights);
        matrix_destroy(mlp->layers[i].gradBiases);
    }

    matrix_destroy(mlp->input);
    matrix_destroy(mlp->inputErrors);
    matrix_destroy(mlp->output);

    free(mlp->layers);
    free(mlp);
}

/* Clears all existing values and sets random weights using the Glorot method. */
void mlp_initialize(MLP *mlp)
{
    for (int i = 0; i <= mlp->depth; i++)
    {        
        double limit = sqrt(6.0 / (double)(mlp->layers[i].weights.rows + mlp->layers[i].weights.columns));
        double *data = mlp->layers[i].weights.data;
        for (int k = 0; k < mlp->layers[i].weights.rows * mlp->layers[i].weights.columns; k++)
            data[k] = deepc_random_double(-limit, limit);
        
        matrix_clear(mlp->layers[i].biases);
        matrix_clear(mlp->layers[i].output);
        matrix_clear(mlp->layers[i].errors);
        matrix_clear(mlp->layers[i].deltas);
        matrix_clear(mlp->layers[i].gradWeights);
        matrix_clear(mlp->layers[i].gradBiases);
    }

    matrix_clear(mlp->input);
    matrix_clear(mlp->inputErrors);
    matrix_clear(mlp->output);
}

/*
   Copies all content from the src to the dst neural network but doesn't change
   its architecture. The caller must ensure identical architectures.
*/
void mlp_copy(MLP *dst, MLP *src)
{
    for (int i = 0; i <= src->depth; i++)
    {
        matrix_copy(dst->layers[i].weights, src->layers[i].weights);
        matrix_copy(dst->layers[i].biases, src->layers[i].biases);
        matrix_copy(dst->layers[i].output, src->layers[i].output);
        matrix_copy(dst->layers[i].errors, src->layers[i].errors);
        matrix_copy(dst->layers[i].deltas, src->layers[i].deltas);
        matrix_copy(dst->layers[i].gradWeights, src->layers[i].gradWeights);
        matrix_copy(dst->layers[i].gradBiases, src->layers[i].gradBiases);
    }
    
    matrix_copy(dst->input, src->input);
    matrix_copy(dst->inputErrors, src->inputErrors);
    matrix_copy(dst->output, src->output);
}

/*
   Performs a feedforward operation with the given input values x and returns the
   output values. All the intermediate layer outputs as well as the final output
   are stored internally to be used during back-propagation.
*/
Matrix mlp_feedforward(MLP *mlp, Matrix x)
{
    matrix_transpose(x, mlp->input);
    Matrix *input = &mlp->input;
    for (int i = 0; i <= mlp->depth; i++)
    {
        matrix_dot(mlp->layers[i].weights, *input, mlp->layers[i].output);
        matrix_add(mlp->layers[i].output, mlp->layers[i].biases);
        matrix_apply(mlp->layers[i].output, mlp->layers[i].activation);
        input = &mlp->layers[i].output;
    }
    matrix_transpose(*input, mlp->output);
    return mlp->output;
}

/*
   Backpropagates the error according to the given true values y and the given
   loss function. The resulting gradients are stored internally. The total error
   over all samples is returned.
*/
double mlp_backpropagate(MLP *mlp, Matrix y, int lossFunctionCode)
{   
    /* Use the loss function to compute the error values. */
    LossFunction lossFunction = getLossFunction(lossFunctionCode);
    double loss = lossFunction(mlp->output, y, mlp->layers[mlp->depth].errors);

    /* Compute the deltas of the last layer. */
    matrix_copy(mlp->layers[mlp->depth].deltas, mlp->output);
    matrix_apply(mlp->layers[mlp->depth].deltas, mlp->layers[mlp->depth].activationDeriv);
    
    /* Compute the error of the previous layer. */
    if (mlp->depth > 0)
        matrix_odot(mlp->layers[mlp->depth].deltas, mlp->layers[mlp->depth].errors);

    /* Propagate the deltas towards the first layer. */
    for (int i = mlp->depth - 1; i >= 0; i--)
    {
        matrix_dot(mlp->layers[i+1].deltas, mlp->layers[i+1].weights, mlp->layers[i].errors);
        matrix_transpose(mlp->layers[i].output, mlp->layers[i].deltas);
        matrix_apply(mlp->layers[i].deltas, mlp->layers[i].activationDeriv);
        matrix_odot(mlp->layers[i].deltas, mlp->layers[i].errors);
    }

    /* Compute the input errors. */
    matrix_dot(mlp->layers[0].deltas, mlp->layers[0].weights, mlp->inputErrors);

    /* Compute the gradients. */
    Matrix *input = &mlp->input;
    for (int i = 0; i <= mlp->depth; i++)
    {
        matrix_dot_transpose(*input, mlp->layers[i].deltas, mlp->layers[i].gradWeights);
        matrix_divide(mlp->layers[i].gradWeights, (double)mlp->batchSize);
        input = &mlp->layers[i].output;

        matrix_sum_rows_transpose(mlp->layers[i].deltas, mlp->layers[i].gradBiases);
        matrix_divide(mlp->layers[i].gradBiases, (double)mlp->batchSize);
    }

    return loss;
}

/*
    Returns the error at the inout layer. This can used to interconnect
    the back-propagation process across different neural networks.
*/
Matrix mlp_get_input_errors(MLP *mlp)
{
    return mlp->inputErrors;
}

/* 
   Performs stohastic gradient descent. The function can be called after the gradients
   have been computed through back-propagation.
*/
void mlp_sgd(MLP *mlp, double lr)
{
    for (int i = 0; i <= mlp->depth; i++)
    {
        matrix_multiply(mlp->layers[i].gradWeights, lr);
        matrix_subtract(mlp->layers[i].weights, mlp->layers[i].gradWeights);

        matrix_multiply(mlp->layers[i].gradBiases, lr);
        matrix_subtract(mlp->layers[i].biases, mlp->layers[i].gradBiases);
    }
}

void mlp_clip_gradients(Matrix gradients, double clipnorm)
{
    double norm = 0;
    for (int i = 0; i < gradients.rows * gradients.columns; i++)
        norm += pow(gradients.data[i], 2);
    norm = sqrt(norm);
    
    if (norm > clipnorm)
        matrix_multiply(gradients, clipnorm / norm);
}

void mlp_sgd_clip(MLP *mlp, double lr, double clipnorm)
{
    for (int i = 0; i <= mlp->depth; i++)
    {
        mlp_clip_gradients(mlp->layers[i].gradWeights, clipnorm);
        matrix_multiply(mlp->layers[i].gradWeights, lr);
        matrix_subtract(mlp->layers[i].weights, mlp->layers[i].gradWeights);

        matrix_multiply(mlp->layers[i].gradBiases, lr);
        matrix_subtract(mlp->layers[i].biases, mlp->layers[i].gradBiases);
    }
}

int mlp_load_weights(MLP *mlp, const char *filename)
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL)
        return -1;
    
    int result = mlp_read_weights(mlp, file);
    fclose(file);
    
    return result;
}

int mlp_read_weights(MLP *mlp, FILE *file)
{
    int rows, columns;
    Matrix matrix;

    for (int i = 0; i <= mlp->depth; i++)
    {
        rows = mlp->layers[i].weights.rows;
        columns = mlp->layers[i].weights.columns;
        
        matrix = matrix_read(file);
        if (matrix.rows != rows || matrix.columns != columns || matrix.data == NULL)
            return -1;
        
        matrix_copy(mlp->layers[i].weights, matrix);
        matrix_destroy(matrix);

        rows = mlp->layers[i].biases.rows;
        columns = mlp->layers[i].biases.columns;

        matrix = matrix_read(file);
        if (matrix.rows != rows || matrix.columns != columns || matrix.data == NULL)
            return -1;
        
        matrix_copy(mlp->layers[i].biases, matrix);
        matrix_destroy(matrix);
    }

    return 0;
}

int mlp_save_weights(MLP *mlp, const char *filename)
{
    FILE *file = fopen(filename, "wb");
    if (file == NULL)
        return -1;
    
    int result = mlp_write_weights(mlp, file);
    fclose(file);
    
    return result;
}

int mlp_write_weights(MLP *mlp, FILE *file)
{
    for (int i = 0; i <= mlp->depth; i++)
    {
        if (matrix_write(mlp->layers[i].weights, file) != 0)
            return -1;

        if (matrix_write(mlp->layers[i].biases, file) != 0)
            return -1;
    }

    return 0;
}