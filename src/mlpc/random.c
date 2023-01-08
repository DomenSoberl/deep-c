#include <stdlib.h>
#include <time.h>
#include "random.h"

void deepc_random_init()
{
    srand((unsigned int)time(NULL));
}

int deepc_random_int(int min, int max)
{
    return rand() % (max - min + 1) + min;
}

double deepc_random_double(double min, double max)
{
    return ((double)rand() / RAND_MAX) * (max - min) + min;
}