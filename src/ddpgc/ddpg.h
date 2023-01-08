/**
 * \file   ddpg.h
 * \author Domen Šoberl
 * \date   January 2023
 * \brief  A C implementation of the DDPG method.
 * 
 * This unit contains the complete DDPGC program code. It defines a single
 * structure called DDPG that represents one instance of the DDPG algorithm.
 */

#include "mlpc.h"

/**
 * This structure contains all the parameters of a DDPG instance, its current
 * internal state and the experience memory. All the memory allocations are
 * made upon creation, so there are no additional allocations during training
 * or acting.
 */
typedef struct DDPG
{
    /**
     * States are defined as tuple of real values. This parameters determines
     * the dimension of the state.
     */
    int stateSize;

    /**
     * Actions are defined as tuples of real values - the output signals. This
     * parameters determines the output dimension.
     */
    int actionSize;

    /**
     * A preallocated array to store and return the action determined by the
     * actor.
     */
    double *action;

    /**
     * A array that stores the user-defined noise level for each action signal.
     */
    double *noise;

    /**
     * The actor neural network.
     */
    MLP *actor;

    /**
     * The critic neural network.
     */
    MLP *critic;

    /**
     * The target actor neural network.
     */
    MLP *actorTarget;

    /**
     * The target critic neural network.
     */
    MLP *criticTarget;

    /**
     * The Adam optimizer for the actor.
     */
    Adam *actorAdam;
    
    /**
     * The Adam optimizer for the critic.
     */
    Adam *criticAdam;

    /**
     * A preallocated matrix that is used as a batch input for either the
     * actor or the target actor.
     */
    Matrix actorInput;

    /**
     * A preallocated matrix that is used as a batch input for either the
     * critic or the target critic.
     */
    Matrix criticInput;

    /**
     * A preallocated matrix that is used as a batch input for the actor error
     * back-propagation.
     */
    Matrix actorErrors;

    /**
     * A preallocated matrix that is used as a batch input for the critic error
     * back-propagation.
     */
    Matrix criticErrors;
    
    /**
     * The size of the batch to train the DDPG.
     */
    int batchSize;

    /**
     * A preallocated array to store randomly chosen indices for the training
     * batch. The size of the array is determined by the `batchSize` value.
     */
    int *batchIndices;

    /**
     * The size of the memory that stores the observations. The memory is
     * preallocated upon DDPG creation.
     */
    int memorySize;

    /**
     * The number of records (observations) contained in the memory.
     */
    int memoryUsed;

    /**
     * The index of the next observation slot to be written in the memory. If
     * the memory is full, the oldest record is overwritten.
     */
    int memoryIdx;

    /**
     * The preallocated memory to store observations. An observations is defined
     * as a tuple (state, action, reward, next state, terminal). The reward
     * and the terminal flag each take up one variable of type double. The size
     * of one observation is therefore equal to `2 * stateSize + actionSize + 2`.
     */
    Matrix memory;

    /**
     * A preallocated array that stores the last observed state.
     */
    double *lastState;

    /**
     * A flag that determines if the `lastState` variable stores a valid state.
     * This flag is reset to 0 initially, and set to 1 after the first
     * observation has been made.
     */
    int lastStateValid;
} DDPG;

/**
 * Initializes the DDPGC library. This function should be called once at the
 * start of the program. It also initializes the MLPC library, so the main
 * program does not have to call the `mlp_init` function.
 */
void ddpg_init();

/**
 * Creates a DDPG on heap. A DDPG created with this function must eventually
 * be destroyed by calling `ddpg_destroy`.
 * 
 * \param stateSize
 * The size (dimension) of states.
 * \param actionSize
 * The size (dimension) of actions.
 * \param noise
 * An array that determines the level of noise for each action signal.
 * \param actorDepth
 * The number of the hidden layers of the actor neural network.
 * \param actorLayers
 * An array determining the size of each actor's layer. Can be NULL if
 * `actorDepth = 0`.
 * \param criticDepth
 * The number of the hidden layers of the critic neural network.
 * \param criticLayers
 * An array determining the size of each critics's layer. Can be NULL if
 * `criticDepth = 0`.
 * \param memorySize
 * The maximum number of observations that the memory can store.
 * \param batchSize
 * The size of the training batch.
 * 
 * \returns The pointer to a newly created and initialized DDPG structure.
 */
DDPG *ddpg_create(
    int stateSize,
    int actionSize,
    double *noise,
    int actorDepth,
    int *actorLayers,
    int criticDepth,
    int *criticLayers,
    int memorySize,
    int batchSize);

/**
 * Frees the memory allocated on the heap by the given DDPG. After being
 * destroyed, the DDPG structure should not be used anymore.
 */
void ddpg_destroy(DDPG *ddpg);

/**
 * At every step of execution, the environmental observation are passed to the
 * DDPG machinery through this function. The previous state is stored internally.
 * 
 * \param ddpg
 * The DDPG instance to pass the observations to.
 * \param action
 * The action that was executed during this step. This is an array of the length
 * determined by the the `actionSize` parameter, given during the creation of
 * `ddpg`.
 * \param reward
 * The reward obtained after the action has been executed.
 * \param state
 * The state after the action has been executed.  This is an array of the length
 * determined by the the `stateSize` parameter, given during the creation of
 * `ddpg`.
 * \param terminal
 * The flag that determines whether this observation is the last one in the
 * current episode. It is set to 1 if it is the terminal state, otherwise to 0.
 */
void ddpg_observe(DDPG *ddpg, double *action, double reward, double *state, int terminal);

/**
 * Returns the action that the given `ddpg` proposes to execute in the given
 * `state`. The `state` is an array of the length determined by the the
 * `stateSize` parameter, given during the creation of `ddpg`.
 * 
 * * \returns An array of output signals of length `actionSize`.
 */
double *ddpg_action(DDPG *ddpg, double *state);

/**
 * Train the given `ddpg` on one randomly selected batch from the memory. The
 * size of the batch is determined by the `batchSize` parameter, given during
 * the creation of `ddpg`.
 */
void ddpg_train(DDPG *ddpg, double gamma);

/**
 * Updates the target actor and critic networks.
 */
void ddpg_update_target_networks(DDPG *ddpg);

/**
 * Signals that a new episode has been started. This invalidates the currently
 * stored state.
 */
void ddpg_new_episode(DDPG *ddpg);

/**
 * Store the trained policy to a file. This saves the weights and biases
 * of the actor and the critic, but no other training data.
 * 
 * \returns 0 if successful, -1 otherwise.
 */
int ddpg_save_policy(DDPG *ddpg, const char *filename);

/**
 * Loads the replay policy from a file. The file does not contain the
 * information on the DDPG architecture, therefore the loading DDPG architecture
 * must be identical to the architecture the file was created with in order
 * for loading to be successful.
 * 
 * \returns 0 if successful, -1 otherwise.
 */
int ddpg_load_policy(DDPG *ddpg, const char *filename);