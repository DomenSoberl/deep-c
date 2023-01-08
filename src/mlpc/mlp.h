/**
 * \file   mlp.h
 * \author Domen Šoberl
 * \date   January 2023
 * \brief  An implementation of Multilayer Perceptrons (MLP)
 * 
 * This unit implements fully connected neural networks (multilayer perceptrons).
 * A MLP is implemented as an array of layers. The input layer is implicit and
 * determined only by the width of the `weights` matrix within the first layer
 * The MLP structure therefore stores all the hidden layers and the output layer.
 * The first hidden layer has index 0, while the output layer has index `depth`.
 *
 * A MLP has a fixed sized dimensions, which is determined upon its creation.
 * This also means a fixed batch size. This enables all the needed matrices
 * to be created initially, without any need for additional memory allocations
 * during MLP processing.
 */

#include <stdio.h>
#include "matrix.h"

/**
 * Definition of a layer within a MLP.
 */
typedef struct Layer
{
    /**
     * Stores the weights between this and the previous layer.
     * Format: (output size / neurons × input size / neurons on previous layer).
     */
    Matrix weights;

    /**
     * Stores the bias for each neuron in a column. For easier computation, the
     * column is repeated horizontally for each sample in the batch.
     * Format: (output size / neurons × batch size)
     */
    Matrix biases;

    /**
     * Stores the layer's output values (after activation). These values are
     * computed by the feedforward method and later used by the back-propagation
     * method, if called.
     * Format (output size / neurons × batch size)
     */
    Matrix output;

    /**
     * The error values computed during back-propagation. The error values of
     * the output layer are set by the loss function. These are then propagated
     * towards the input layer by the back-propagation method.
     * Format: (batch size × output size / neurons)
     */
    Matrix errors;

    /**
     * Local gradients computed during back-propagation, based on the errors.
     * Format: (batch size × output size / neurons)
     */
    Matrix deltas;

    /**
     * Weight gradients computed during back-propagation, based on the deltas.
     * Format: (output size / neurons × input size / neurons on previous layer)
     */
    Matrix gradWeights;

    /**
     * Bias gradients computed during back-propagation, based on the deltas.
     * The column is repeated for each sample in the batch.
     * Format: (output size / neurons × batch size)
     */
    Matrix gradBiases;

    /**
     * The activation function that is applied during feedforward to the output
     * of every neuron on this layer.
     */
    ActivationFunction activation;

    /**
     * The derivative of the activation function that is applied during
     * back-propagation to the output of every neuron on this layer.
     */
    ActivationFunction activationDeriv;
} Layer;

/**
 * Definition of a MLP.
 */
typedef struct MLP
{
    /**
     * The number of the hidden layers.
     */
    int depth;

    /**
     * The number of samples within a batch. This value is constant and a
     * batch of this size must always be provided to the feedforward and the
     * back-propagation operations.
     */
    int batchSize;

    /**
     * The array of layers, which starts with the first hidden layer and ends
     * with the output layer at index `depth`.
     */
    Layer *layers;

    /**
     * Because the input layer is not stored within the MLP structure, the
     * `input` matrix stores the input values. To comply with the matrix
     * operations between the MLP layer, this matrix is the transposed matrix
     * of the actual input.
     * Format: (input size / neurons × batch size)
     */
    Matrix input;

    /**
     * When errors are propagated backwards, they eventually reach the input
     * layer. But because there is no input layer stored within the MLP
     * structure, the errors at the input layer are stored in the `inputErrors`
     * matrix. These are useful when we want to propagate errors across multiple
     * connected MLPs.
     * Format: (batch size × input size / neurons)
     */
    Matrix inputErrors;

    /**
     * The output of the last layer, but transposed to match the format of the
     * input batch. This matrix is returned by the `feedforward` function.
     */
    Matrix output;
} MLP;

/**
 * Initializes the MLPC library. This function should be called once at the
 * start of the program.
 */
void mlp_init();

/**
 * Creates a MLP on the heap. A MLP created with this function must eventually
 * be destroyed by calling `mlp_destroy`.
 * 
 * \param inputSize
 * The number of input values (the number of neurons on the input layer).
 * \param outputSize
 * The number of output values (the number of neurons on the output layer).
 * \param depth
 * The number of hidden layers.
 * \param hiddenLayerSizes
 * The size of each hidden layer, given as an array of integers. Can be NULL if
 * `depth = 0`.
 * \param hiddenLayerActivation
 * The code of the activation function to be used on the hidden layers.
 * \param outputLayerActivation
 * The code of the activation function to be used on the output layer.
 * \param batchSize
 * The number of samples that this MLP processes at once.
 * 
 * \returns The newly created and initialized MLP structure.
 */
MLP *mlp_create(
    int inputSize,
    int outputSize,
    int depth,
    int *hiddenLayerSizes,
    int hiddenLayerActivation,
    int outputLayerActivation,
    int batchSize);

/**
 * Creates a new MLP that is a clone of the given MLP. All the content from the
 * given MLP is copied to the newly created MLP. A MLP created using this
 * function must eventually be destroyed by calling `mlp_destroy`.
 * 
 * \returns The newly created and initialized MLP structure.
 */
MLP *mlp_clone(MLP *mlp);

/**
 * Frees the memory allocated on the heap by the given MLP. After being
 * destroyed, the MLP structure should not be used anymore.
 */
void mlp_destroy(MLP *mlp);

/**
 * Initializes the weights and biases of the given MLP with random values using
 * the Glorot method. All other variables within the structure are initialized
 * with zero values. A MLP created with the `mlp_create` function is
 * autmatically initialized, so this function needs to be called only if we
 * want to completely reset a MLP.
 */
void mlp_initialize(MLP *mlp);

/**
 * Copies all the contents from the `src` MLP to the `dst` MLP. Both MLPs must
 * be of identical architecture. It is easiest and safest to use this function
 * on MLP clones.
 */
void mlp_copy(MLP *dst, MLP *src);

/**
 * Performs the feedforward operation on the given batch `x`. The shape of
 * matrix `x` must conform to the input shape of the MLP (batch size × 
 * input size).
 *
 * The outputs are computed as follows:
 * 
 *     output[i] = activation(weights[i] x output[i-1] + biases[i])
 * 
 * \returns The predicted `y` values. The returned matrix must not be destroyed
 * by the caller.
 */
Matrix mlp_feedforward(MLP *mlp, Matrix x);

/**
 * Performs the back-propagation operation using the errors obtained from the
 * given true values `y` and the loss function given with the integer code
 * `lossFunctionCode`. The predicted outputs are stored internally after
 * the last feedforward call. The function computes the error values and the
 * delta values (local gradient) as well as the weight and bias gradients
 * on each layer.
 * 
 * The values are computed as follows:
 * 
 *     deltas[depth] = activationDeriv(error) * activationDeriv(output[i])^T
 *     deltas[i] = (deltas[i+1] x weights[i+1]) * activationDeriv(output[i])^T
 *     gradWeights = (1/batch size) * (output[i-1] x deltas[i])^T  
 *     gradBiases = (1/batch size) * ([1,1,...,1] x deltas[i]))^T x [1,1,...,1]
 * 
 * \returns The mean error computed by the loss function.
 */
double mlp_backpropagate(MLP *mlp, Matrix y, int lossFunctionCode);

/**
 * Returns the error values at the input level. This is useful when performing
 * back-propagation throughout multiple connected neural networks.
 * 
 * \returns The matrix containing the errors. The returned matrix should not
 * ne destroyed by the caller.
 */
Matrix mlp_get_input_errors(MLP *mlp);

/**
 * Performs stohastic gradient descent with the given learning rate `lr`. This
 * Function can be called after `mlp_backpropagate`, which computes and
 * stores internally the needed gradients.
 * 
 * New weights and biases are computed as follows:
 * 
 *     weights[i] -= (learning rate) * gradWeights[i]
 *     biases[i] -= (learning rate) * gradBiases[i]
 */
void mlp_sgd(MLP *mlp, double lr);

/**
 * Stohastic gradient descent with the additional gradient clipping mechanism.
 */
void mlp_sgd_clip(MLP *mlp, double lr, double clipnorm);

/**
 * Loads weights and biases from a file.
 * 
 * \returns 0 if successful, -1 otherwise.
 */
int mlp_load_weights(MLP *mlp, const char *filename);

/**
 * Loads weights and biases from a binary stream.
 * 
 * \returns 0 if successful, -1 otherwise.
 */
int mlp_read_weights(MLP *mlp, FILE *file);

/**
 * Saves weights and biases to a file.
 * 
 * \returns 0 if successful, -1 otherwise.
 */
int mlp_save_weights(MLP *mlp, const char *filename);

/**
 * Saves weights and biases to a binary stream.
 * 
 * \returns 0 if successful, -1 otherwise.
 */
int mlp_write_weights(MLP *mlp, FILE *file);