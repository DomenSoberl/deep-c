/**
 * \file   pendulum.c
 * \author Domen Šoberl
 * \date   January 2023
 * \brief  Solving the pendulum swing up problem.
 * 
 * This is an example of how to use the DDPGC library, demonstrated on the
 * well-known pendulum swing up problem, which can also be found in the
 * Gymnasium reinforcement learning collection for Python. This program does
 * not do any pendulum rendering, just computes the states.
 */

#include <math.h>
#include <stdio.h>
#include "ddpgc.h"

#define PI 3.14159265358979323846

#define MAX_SPEED 8.0
#define DT 0.05
#define G 9.81
#define MASS 1.0
#define LENGTH 1.0

#define EPISODE_LENGTH 200
#define EPISODE_COUNT 100
#define STARTING_EPISODES 3

/* We store the states {theta, thetadot} within an array of type double. */
double state[2] = {0, 0};

/* Similar for actions, but it is an array with only one element. */
double action[1] = {0};

/* Simulate the motion of the pendulum and return the reward of the current state. */
double pendulum_step(double *state, double action)
{
    /* Get the current state. */
    double theta = state[0];
    double thetadot = state[1];

    /* Compute the cost of the current state. */
    double cost = pow(theta, 2) + 0.1 * pow(thetadot, 2) + 0.001 * pow(action, 2);

    /* The new pendulum velocity. */
    thetadot += (3 * G / (2 * LENGTH) * sin(theta) + 3.0 / (MASS * pow(LENGTH, 2)) * action) * DT;
    
    /* Clip the velocity to [-MAX_SPEED, MAX_SPEED]. */
    if (thetadot < -MAX_SPEED)
        thetadot = -MAX_SPEED;
    if (thetadot > MAX_SPEED)
        thetadot = MAX_SPEED;

    /* The new pendulum position. */
    theta = theta + thetadot * DT;

    /* Clip the position to [-2*PI, 2*PI]. */
    if (theta > PI)
        theta -= 2 * PI;
    if (theta < -PI)
        theta += 2 * PI;

    /* Store the new state. */
    state[0] = theta;
    state[1] = thetadot;

    /* The reward is the negative cost. */
    return -cost;
}

int main()
{
    /* Initialize the DDPGC library. */
    ddpg_init();

    /*
       We will use a DDPG method with the following properties:
        - States have 2 values, actions have 1 value.
        - Actions have 0.01 noise.
        - The actor and the critic bot have 2 hidden layers with 128 and 64 neurons.
        - Memory capacity is 100K observations.
        - Train on batches of 32 samples.
    */
    int layers[2] = {128, 64};
    double noise[1] = {0.01};
    DDPG *ddpg = ddpg_create(2, 1, noise, 2, layers, 2, layers, 100000, 32);

    /* Try to load the pre-trained model. */
    if (ddpg_load_policy(ddpg, "pendulum.ddpg") == 0)
        printf("Loaded the pre-trained model.\n");
    else
        printf("No pre-trained model. Training from scratch.\n");

    /* Execute several episodes. */
    for (int episode = 0; episode < EPISODE_COUNT; episode++)
    {
        /* Measure the average episode reward. */
        double episodeReward = 0;

        /* Reset the pendulum state. */
        state[0] = deepc_random_double(-PI, PI);
        state[1] = 0;

        /* Start a new DDPG episode. */
        ddpg_new_episode(ddpg);

        /* Each episode is composed of a fixed number of steps. */
        for (int step = 0; step < EPISODE_LENGTH; step++)
        {
            /* For the first few episodes only do random exploration. */
            if (episode < STARTING_EPISODES)
                action[0] = deepc_random_double(-1, 1);
            else
                action[0] = *ddpg_action(ddpg, state);

            /* Simulate one step of pendulum motion. Scale the action to the
            [-2, 2] interval, which is the action range for this domain. */
            double reward = pendulum_step(state, 2 * action[0]);

            /* Record the received reward. */
            episodeReward += reward;

            /* Let the DDPG algorithm observe the action that we took, the
               reward we received and the consequent state. We do not treat
               any state as terminal, because this domain ends the episodes
               in arbitrary states, and not in fail states or states that
               can actually have no consequent states. */
            ddpg_observe(ddpg, action, reward, state, 0);
            
            /* If we are not in the exploration phase, train the DDPG. */
            if (episode >= STARTING_EPISODES)
                ddpg_train(ddpg, 0.99);
        }

        /* Update the target actor and the target critic after each episode. */
        ddpg_update_target_networks(ddpg);
        
        /* Print out the average episode reward. */
        printf("%d %f\n", episode, episodeReward / EPISODE_LENGTH);
    }

    /* Save the trained model. */
    if (ddpg_save_policy(ddpg, "pendulum.ddpg") == 0)
        printf("Trained model saved.\n");
    else
        printf("Could not save the trained model.\n");

    /* Destroy the DDPG structure. */
    ddpg_destroy(ddpg);

    return 0;
}