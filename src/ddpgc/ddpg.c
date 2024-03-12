#include <stdlib.h>
#include <malloc.h>
#include "ddpg.h"

#include <stdio.h>

void ddpg_init()
{
    mlp_init();
}

DDPG *ddpg_create(
    int stateSize,
    int actionSize,
    double *noise,
    int actorDepth,
    int *actorLayers,
    int criticDepth,
    int *criticLayers,
    int memorySize,
    int batchSize)
{
    DDPG *ddpg = malloc(sizeof(DDPG));
    ddpg->stateSize = stateSize;
    ddpg->actionSize = actionSize;

    /* Action returned. */
    ddpg->action = malloc(actionSize * sizeof(double));
    
    /* If noise is NULL, no noise is applied to actions. */
    ddpg->noise = NULL;
    if (noise != NULL)
    {
        ddpg->noise = malloc(actionSize * sizeof(double));
        for (int i = 0; i < actionSize; i++)
            ddpg->noise[i] = noise[i];
    }
    
    /* The MLPC library is used to construct the actor and the critic. */
    ddpg->actor = mlp_create(stateSize, actionSize, actorDepth, actorLayers, ACTIVATION_RELU, ACTIVATION_TANH, batchSize);
    ddpg->critic = mlp_create(actionSize + stateSize, 1, criticDepth, criticLayers, ACTIVATION_RELU, ACTIVATION_LINEAR, batchSize);
    ddpg->actorTarget = mlp_clone(ddpg->actor);
    ddpg->criticTarget = mlp_clone(ddpg->critic);

    /* Initialize the Adam optimizers. */
    ddpg->actorAdam = adam_create(ddpg->actor);
    ddpg->criticAdam = adam_create(ddpg->critic);

    /* Initialize matrices for actors and critics. */
    ddpg->actorInput = matrix_create(batchSize, stateSize);
    ddpg->criticInput = matrix_create(batchSize, actionSize + stateSize);
    ddpg->actorErrors = matrix_create(batchSize, actionSize);
    ddpg->criticErrors = matrix_create(batchSize, 1);

    /* Initialize batch capacity. */
    ddpg->batchSize = batchSize;
    ddpg->batchIndices = malloc(batchSize * sizeof(int));

    /* The memory stores the current state, action, reward, next state, and the terminal flag. */
    ddpg->memory = matrix_create(memorySize, actionSize + 2 * stateSize + 2);
    ddpg->memorySize = memorySize;  
    ddpg->memoryUsed = 0;
    ddpg->memoryIdx = 0;

    /* Last observed state. */
    ddpg->lastState = malloc(ddpg->stateSize * sizeof(double));
    ddpg->lastStateValid = 0;

    return ddpg;
}

void ddpg_destroy(DDPG *ddpg)
{
    free(ddpg->action);

    mlp_destroy(ddpg->actor);
    mlp_destroy(ddpg->critic);
    mlp_destroy(ddpg->actorTarget);
    mlp_destroy(ddpg->criticTarget);

    adam_destroy(ddpg->actorAdam);
    adam_destroy(ddpg->criticAdam);

    matrix_destroy(ddpg->actorInput);
    matrix_destroy(ddpg->criticInput);
    matrix_destroy(ddpg->actorErrors);
    matrix_destroy(ddpg->criticErrors);

    if (ddpg->noise != NULL)
        free(ddpg->noise);

    free(ddpg->batchIndices);
    matrix_destroy(ddpg->memory);
    
    free(ddpg->lastState);

    free(ddpg);
}

void ddpg_data_copy(double *dst, double *src, int length)
{
    for (int i = 0; i < length; i++)
        *(dst++) = *(src++);
}

void ddpg_observe(DDPG *ddpg, double *action, double reward, double *state, int terminal)
{
    /* If no state has yet been observed, just store the state. */
    if (!ddpg->lastStateValid)
    {
        ddpg_data_copy(ddpg->lastState, state, ddpg->stateSize);
        ddpg->lastStateValid = 1;
        return;
    }

    /* Copy the given data to the observation memory. */
    int col = 0;
    ddpg_data_copy(&MATRIX(ddpg->memory, ddpg->memoryIdx, 0), ddpg->lastState, ddpg->stateSize);
    ddpg_data_copy(&MATRIX(ddpg->memory, ddpg->memoryIdx, (col += ddpg->stateSize)), action, ddpg->actionSize);
    MATRIX(ddpg->memory, ddpg->memoryIdx, (col += ddpg->actionSize)) = reward;
    ddpg_data_copy(&MATRIX(ddpg->memory, ddpg->memoryIdx, (col += 1)), state, ddpg->stateSize);
    MATRIX(ddpg->memory, ddpg->memoryIdx, (col + ddpg->stateSize)) = (terminal > 0 ? 1.0 : 0.0);

    /* Store the given state as the last observed state. */
    ddpg_data_copy(ddpg->lastState, state, ddpg->stateSize);

    /* Increase the record index and memory size. */
    ddpg->memoryIdx = (ddpg->memoryIdx + 1) % ddpg->memorySize;
    if (ddpg->memoryUsed < ddpg->memorySize)
        ddpg->memoryUsed++;
}

double *ddpg_action(DDPG *ddpg, double *state)
{
    /* The actor expects a batch, but we only need to process one instance. We
       use only the first sample in the batch and set the rest to 0. */
    matrix_clear(ddpg->actorInput);
    ddpg_data_copy(ddpg->actorInput.data, state, ddpg->stateSize);
    Matrix action = mlp_feedforward(ddpg->actor, ddpg->actorInput);

    /* Copy the resulting action to the DDPG structure. */
    for (int i = 0; i < ddpg->actionSize; i++)
    {
        ddpg->action[i] = MATRIX(action, 0, i);

        /* If action noise is set, apply it to the individual output signals. */
        if (ddpg->noise != NULL)
        {
            ddpg->action[i] += deepc_random_double(-ddpg->noise[i], ddpg->noise[i]);

            /* Clip the action to interval [-1, 1]. */
            if (ddpg->action[i] > 1)
                ddpg->action[i] = 1;
            else if (ddpg->action[i] < -1)
                ddpg->action[i] = -1;
        }
    }

    return ddpg->action;
}

void ddpg_train(DDPG *ddpg, double gamma)
{
    /* If not enough samples in memory, do nothing. */
    if (ddpg->memoryUsed < ddpg->batchSize)
        return;

    /* Select a random batch. */
    for (int i = 0; i < ddpg->batchSize; i++)
        ddpg->batchIndices[i] = deepc_random_int(0, ddpg->memoryUsed - 1);

    /* Train the actor. */
    
    /* Set the input states for the actor. */
    for (int i = 0; i < ddpg->batchSize; i++)
        ddpg_data_copy(&MATRIX(ddpg->actorInput, i, 0), &MATRIX(ddpg->memory, ddpg->batchIndices[i], 0), ddpg->stateSize);

    /* Get the proposed actions for the input states. */
    Matrix proposedActions = mlp_feedforward(ddpg->actor, ddpg->actorInput);

    /* Concatenate the proposed actions with batch states. */
    for (int i = 0; i < ddpg->batchSize; i++)
    {
        ddpg_data_copy(&MATRIX(ddpg->criticInput, i, 0), &MATRIX(proposedActions, i, 0), ddpg->actionSize);
        ddpg_data_copy(&MATRIX(ddpg->criticInput, i, ddpg->actionSize), &MATRIX(ddpg->memory, ddpg->batchIndices[i], 0), ddpg->stateSize);
    }

    /* Process the proposed actions through the critic. */
    mlp_feedforward(ddpg->critic, ddpg->criticInput);

    /* Back-propagate the negative gradient through the critic. */
    matrix_fill(ddpg->criticErrors, -1);
    mlp_backpropagate(ddpg->critic, ddpg->criticErrors, LOSS_NONE);

    /* Get the critic errors of the first layer and extract only those that correspond to actions. */
    Matrix errors = mlp_get_input_errors(ddpg->critic);
    for (int i = 0; i < ddpg->batchSize; i++)
        ddpg_data_copy(&MATRIX(ddpg->actorErrors, i, 0), &MATRIX(errors, i, 0), ddpg->actionSize);

    /* Continue the back-propagation through the actor. */
    mlp_backpropagate(ddpg->actor, ddpg->actorErrors, LOSS_NONE);

    /* Optimize the actor */
    adam_optimize(ddpg->actor, ddpg->actorAdam);

    /* Train the critic. */

    /* Feed the batch actions and states to the critic. */
    for (int i = 0; i < ddpg->batchSize; i++)
    {
        ddpg_data_copy(&MATRIX(ddpg->criticInput, i, 0), &MATRIX(ddpg->memory, ddpg->batchIndices[i], ddpg->stateSize), ddpg->actionSize);
        ddpg_data_copy(&MATRIX(ddpg->criticInput, i, ddpg->actionSize), &MATRIX(ddpg->memory, ddpg->batchIndices[i], 0), ddpg->stateSize);
    }

    Matrix criticOutput = mlp_feedforward(ddpg->critic, ddpg->criticInput);

    /* Feed the next state batch to the target actor. */
    for (int i = 0; i < ddpg->batchSize; i++)
        ddpg_data_copy(&MATRIX(ddpg->actorInput, i, 0), &MATRIX(ddpg->memory, ddpg->batchIndices[i], (ddpg->stateSize + ddpg->actionSize + 1)), ddpg->stateSize);
    
    Matrix actorTargetOutput = mlp_feedforward(ddpg->actorTarget, ddpg->actorInput);

    /* Concatenate target actions with the next state batch and feed it to the target critic. */
    for (int i = 0; i < ddpg->batchSize; i++)
    {
        ddpg_data_copy(&MATRIX(ddpg->criticInput, i, 0), &MATRIX(actorTargetOutput, i, 0), ddpg->actionSize);
        ddpg_data_copy(&MATRIX(ddpg->criticInput, i, ddpg->actionSize), &MATRIX(ddpg->memory, ddpg->batchIndices[i], (ddpg->stateSize + ddpg->actionSize + 1)), ddpg->stateSize);
    }

    Matrix CriticTargetOutput = mlp_feedforward(ddpg->criticTarget, ddpg->criticInput);

    /* Compute the critic errors using the Bellman equation. */
    for (int i = 0; i < ddpg->batchSize; i++)
    {
        double reward = MATRIX(ddpg->memory, ddpg->batchIndices[i], (ddpg->stateSize + ddpg->actionSize));
        double terminal = MATRIX(ddpg->memory, ddpg->batchIndices[i], (2 * ddpg->stateSize + ddpg->actionSize + 1));

        if (terminal > 0)
            MATRIX(ddpg->criticErrors, i, 0) = MATRIX(criticOutput, i, 0);
        else
            MATRIX(ddpg->criticErrors, i, 0) = MATRIX(criticOutput, i, 0) - (reward + gamma * MATRIX(CriticTargetOutput, i, 0));
    }

    /* Backpropagate critic errors. */
    mlp_backpropagate(ddpg->critic, ddpg->criticErrors, LOSS_NONE);

    /* Optimize the critic. */
    adam_optimize(ddpg->critic, ddpg->criticAdam);
}

void ddpg_update_target_networks(DDPG *ddpg)
{
    mlp_copy(ddpg->actorTarget, ddpg->actor);
    mlp_copy(ddpg->criticTarget, ddpg->critic);
}

void ddpg_new_episode(DDPG *ddpg)
{
    ddpg->lastStateValid = 0;
}

int ddpg_save_policy(DDPG *ddpg, const char *filename)
{
    FILE *file = fopen(filename, "wb");
    if (file == NULL)
        return -1;

    if (mlp_write_weights(ddpg->actor, file) != 0)
    {
        fclose(file);
        return -1;
    }

    if (mlp_write_weights(ddpg->critic, file) != 0)
    {
        fclose(file);
        return -1;
    }

    fclose(file);
    return 0;
}

int ddpg_load_policy(DDPG *ddpg, const char *filename)
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL)
        return -1;

    if (mlp_read_weights(ddpg->actor, file) != 0)
    {
        fclose(file);
        return -1;
    }
    
    if (mlp_read_weights(ddpg->critic, file) != 0)
    {
        fclose(file);
        return -1;
    }

    fclose(file);
    return 0;
}
