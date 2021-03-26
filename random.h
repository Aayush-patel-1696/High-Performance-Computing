#include <random>

std::default_random_engine generator;
std::uniform_real_distribution<double> distribution(0,1);
#pragma omp declare simd
double randk()
{
    return distribution(generator);
}