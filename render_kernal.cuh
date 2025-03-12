#pragma once
#include "vec3.h"
#include "hittable.h"
#include "point_light.h"
#include <cstdint>
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

__global__ void render_kernel(
    uint8_t* d_pixels,
    int image_width, int image_height,
    int samples_per_pixel, int max_depth,
    const hittable* world, const point_light* lights, int light_count,
    point3 center,
    double pixel_samples_scale,
    point3 pixel00_loc, vec3 pixel_delta_u, vec3 pixel_delta_v,
    double defocus_angle, vec3 defocus_disk_u, vec3 defocus_disk_v,
    color background
);
