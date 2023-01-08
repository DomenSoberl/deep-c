/**
 * \file   random.h
 * \author Domen Šoberl
 * \date   January 2023
 * \brief  Random number generators
 * 
 * This unit defines random number generators that are used by the rest of the
 * library. They are also exposed to the outside user.
 */

/**
 * Initializes the seed. This is done when initializing the library.
 */
void deepc_random_init();

/**
 * \returns A random number of type int between ´min´ and ´max´, both extremes
 * inclusive.
 */
int deepc_random_int(int min, int max);

/**
 * \returns A random number of type double between ´min´ and ´max´.
 */
double deepc_random_double(double min, double max);