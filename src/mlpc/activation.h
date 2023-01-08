/**
 * \file   activation.h
 * \author Domen Šoberl
 * \date   January 2023
 * \brief  Activation functions
 *
 * This module defines several activation functions and their derivatives that
 * can be used with neural networks. Activation functions and their derivatives
 * are defined as C functions of type `double f(double)`. They are not visible
 * outside the module, but can be obtained as the `ActivationFunction` type of
 * pointers.
 * 
 * The derivative functions expect as the input the output of their
 * corresponding activation function. This requires less computation and
 * less storage in the neural network. Luckily, all derivative functions use
 * here can be defined to work with activation outputs.
 */

/**
 * When not using an activation function, provide the ACTIVATION_NONE code.
 * This is equivalent to using the linear activation function f(x) = x.
 */
#define ACTIVATION_NONE    0

/** The linear activation function integer code. */
#define ACTIVATION_LINEAR  0

/** The sigmoid activation function integer code. */
#define ACTIVATION_SIGMOID 1

/** The tanh activation function integer code. */
#define ACTIVATION_TANH    2

/** The ReLu activation function integer code. */
#define ACTIVATION_RELU    3

/**
 * The definition of a pointer to an activation function.
 */
typedef double (*ActivationFunction)(double);

/**
 * \returns the pointer to the activation function that corresponds to the
 * given integer `code`.
 */
ActivationFunction getActivationFunction(int code);

/**
 * \returns the pointer to the derivative of the activation function that
 * corresponds to the given integer `code`.
 */
ActivationFunction getActivationFunctionDeriv(int code);