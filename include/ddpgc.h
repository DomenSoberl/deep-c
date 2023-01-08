/**
 * \file   ddpgc.h
 * \author Domen Šoberl
 * \date   January 2023
 * \brief  The public header that exposes the DDPGC API.
 * 
 * This library implements the DDPG (Deep Deterministic Policy Gradient) method
 * in pure C programming language. It is suitable for the use in embedded
 * systems or any AI related project written in C/C++. It uses the MLPC
 * precompiled static library lib/mlpc to implement the actor and the critic
 * neural networks.
 * 
 * This header is intended to be used with the precompiled lib/ddpgc static
 * library. Documentation for the exposed functions can be found within the
 * DDPGC source code.
 */

typedef struct DDPG DDPG;

void ddpg_init();

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

void ddpg_destroy(DDPG *ddpg);
void ddpg_observe(DDPG *ddpg, double *action, double reward, double *state, int terminal);
double *ddpg_action(DDPG *ddpg, double *state);
void ddpg_train(DDPG *ddpg, double gamma);
void ddpg_update_target_networks(DDPG *ddpg);
void ddpg_new_episode(DDPG *ddpg);
int ddpg_save_policy(DDPG *ddpg, const char *filename);
int ddpg_load_policy(DDPG *ddpg, const char *filename);

int deepc_random_int(int min, int max);
double deepc_random_double(double min, double max);