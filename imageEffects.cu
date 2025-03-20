#include "imageEffects.cuh"

#include <device_launch_parameters.h>
#include <curand_kernel.h>

__global__ void generateBinaryNoise(unsigned char* __restrict__ img, const int nPixels, size_t seed) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= nPixels) return;

    int idx = pIdx * 3;

    // Use a more unique seed by incorporating blockIdx.x and threadIdx.x
    curandState state;
    curand_init(seed, pIdx, 0, &state);

    // Generate a random binary value (0 or 255)
    unsigned char c = (curand(&state) & 1) * 255;

    // Set all three channels (RGB) to the same value
    img[idx] = c;
    img[++idx] = c;
    img[++idx] = c;
}
