#ifndef POINT_LIGHT
#define POINT_LIGHT

#include "vec3.h"
#include <cuda_runtime.h>

using color = vec3;

class point_light {
public:
    // Constructor to initialize the position, intensity, and size
    __host__ __device__ point_light(point3 position, color intensity, double size)
        : position(position), intensity(intensity), size(size) {}

    // Getter for position (device-compatible)
    __host__ __device__ point3 get_position() const { return position; }

    // Getter for intensity (device-compatible)
    __host__ __device__ color get_intensity() const { return intensity; }

    // Getter for size (device-compatible)
    __host__ __device__ double get_size() const { return size; }

private:
    point3 position;  // Position of the light
    color intensity;  // Intensity of the light
    double size;      // Size of the light
};

#endif // !POINT_LIGHT
