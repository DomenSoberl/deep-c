#include <math.h>
#include "activation.h"

double activation_linear(double x)
{
    return x;
}

double activation_linearDeriv(double y)
{
    return 1;
}

double activation_sigmoid(double x)
{
    if (x >= 0)
        return 1.0 / (1 + exp(-x));
    else
        return 1.0 - (1.0 / 1 + exp(x));
}

double activation_sigmoidDeriv(double y)
{
    return y * (1 - y);
}

double activation_tanh(double x)
{
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double activation_tanhDeriv(double y)
{
    return 1 - y*y;
}

double activation_relu(double x)
{
    if (x >= 0)
        return x;
    else
        return 0;
}

double activation_reluDeriv(double y)
{
    if (y > 0)
        return 1;
    else
        return 0;
}

ActivationFunction getActivationFunction(int code)
{
    switch (code)
    {
        case ACTIVATION_SIGMOID:
            return activation_sigmoid;
        case ACTIVATION_TANH:
            return activation_tanh;
        case ACTIVATION_RELU:
            return activation_relu;
        default:
            return activation_linear;
    }
}

ActivationFunction getActivationFunctionDeriv(int code)
{
    switch (code)
    {
        case ACTIVATION_SIGMOID:
            return activation_sigmoidDeriv;
        case ACTIVATION_TANH:
            return activation_tanhDeriv;
        case ACTIVATION_RELU:
            return activation_reluDeriv;
        default:
            return activation_linearDeriv;
    }
}