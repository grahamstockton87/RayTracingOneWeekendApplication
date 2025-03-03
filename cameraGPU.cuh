#ifndef CAMERA_H
#define CAMERA_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "hittable.h"
#include "vec3.h"
#include "ray.h"
#include "interval.h"
#include "material.h"
#include "point_light.h"

#include <string>
#include <future>
#include <algorithm>
#include <fstream>
#include <memory>
#include <iostream>
#include <algorithm>  // For std::max

#define CHANNEL_NUM 3
#include "stb_image_write.h"

using color = vec3;

__device__ inline double linear_to_gamma(double linear_component) {
    if (linear_component > 0) {
        return sqrt(linear_component);
    }
    return 0;
}

class camera {
public:
    const int image_width = 1024;
    const double aspect_ratio = 16.0 / 9.0;
    const char* image_name = "Default Image";
    int samples_per_pixel = 10;
    int max_depth = 10;
    color background = vec3(0, 0, 0);

    double vfov = 90; // field of view
    point3 lookfrom = point3(0, 0, 0); // point cam looking from
    point3 lookat = point3(0, 0, -1); // point cam looking at 
    vec3 vup = vec3(0, 1, 0); // relative up direction

    double defocus_angle = 0;  // Variation angle of rays through each pixel
    double focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus

    __global__ void render(const hittable& world, std::vector<point_light>& lights) {
        initialize();

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        // Allocate GPU memory
        uint8_t* d_pixels;
        cudaMalloc(&d_pixels, image_width * image_height * CHANNEL_NUM * sizeof(uint8_t));

        // Launch CUDA kernel to compute pixel colors
        dim3 block_size(16, 16); // Use a 16x16 block size
        dim3 grid_size((image_width + block_size.x - 1) / block_size.x, (image_height + block_size.y - 1) / block_size.y);

        render_kernel<<<grid_size, block_size >> > (d_pixels, image_width, image_height, samples_per_pixel, max_depth, world, lights, center, pixel_samples_scale, pixel00_loc, pixel_delta_u, pixel_delta_v, defocus_angle, defocus_disk_u, defocus_disk_v);


        // Check for errors in kernel launch
        cudaDeviceSynchronize();

        // Retrieve the image from GPU to CPU memory
        uint8_t* pixels = new uint8_t[image_width * image_height * CHANNEL_NUM];
        cudaMemcpy(pixels, d_pixels, image_width * image_height * CHANNEL_NUM * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        // Free GPU memory
        cudaFree(d_pixels);

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        double time_taken = (std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
        std::cout << "\nDone rendering " << image_name << " in " << time_taken / 60 << " seconds" << std::endl;

        stbi_write_png(image_name, image_width, image_height, CHANNEL_NUM, pixels, image_width * CHANNEL_NUM);
        delete[] pixels;
    }

private:
    int image_height;
    double pixel_samples_scale;
    point3 center;
    point3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    vec3 u, v, w;
    vec3 defocus_disk_u;
    vec3 defocus_disk_v;

    void initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        pixel_samples_scale = 1.0 / samples_per_pixel;
        center = lookfrom;

        auto theta = degrees_to_radians(vfov);
        auto h = std::tan(theta / 2);
        auto viewport_height = 2 * h * focus_dist;
        auto viewport_width = viewport_height * (double(image_width) / image_height);

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        auto viewport_u = viewport_width * u;
        auto viewport_v = viewport_height * -v;

        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        auto viewport_upper_left = center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    __device__ ray get_ray(int i, int j, point3 pixel00_loc, vec3 pixel_delta_u, vec3 pixel_delta_v) const {
        auto offset = sample_square();
        auto pixel_sample = pixel00_loc + ((i + offset.x()) * pixel_delta_u) + ((j + offset.y()) * pixel_delta_v);
        auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
        auto ray_direction = pixel_sample - ray_origin;
        auto ray_time = random_double();

        return ray(ray_origin, ray_direction, ray_time);
    }
    __device__ vec3 sample_square() const {
        //return vector between point in [-0.5,-0.5] - [0.5, 0.5] in the unit square;
        return vec3(random_double() - 0.5, random_double() - 0.5, 0);
    }
    __device__ point3 defocus_disk_sample() const {
        // Returns a random point in the camera defocus disk.
        auto p = random_in_unit_disk();
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }
    __device__ color ray_color(const ray& r, int depth, const hittable& world, std::vector<point_light>& lights) {
        if (depth <= 0)
            return color(0, 0, 0);

        hit_record rec;

        if (!world.hit(r, interval(0.001, infinity), rec)) {
            return background;
        }

        color color_from_emission = rec.mat->emitted(rec.u, rec.v, rec.p);

        ray scattered;
        color attenuation;

        if (!rec.mat->scatter(r, rec, attenuation, scattered)) {
            return color_from_emission;
        }

        color lighting = attenuation * get_lighting(rec.p, rec.normal, lights);
        color color_from_scatter = attenuation * ray_color(scattered, depth - 1, world, lights);

        return color_from_emission + lighting + color_from_scatter;
    }

    __device__ color get_lighting(const point3& p, const vec3& normal, std::vector<point_light>& lights) {
        color result(0, 0, 0);

        for (const auto& light : lights) {
            vec3 light_dir = light.get_position() - p;
            double distance_squared = light_dir.length_squared();
            light_dir = unit_vector(light_dir);

            double diffuse = std::max(dot(normal, light_dir), 0.0);

            double size_factor = light.get_size();
            double radius_effect = size_factor * 0.1;

            if (distance_squared <= size_factor * size_factor) {
                result += light.get_intensity() * diffuse;
            }
            else {
                double attenuation = 1.0 / (distance_squared + radius_effect);
                color intensity = light.get_intensity() * attenuation;
                result += intensity * diffuse;
            }
        }

        return result;
    }
    // CUDA Kernel for rendering
    __global__ void render_kernel(uint8_t* d_pixels, int image_width, int image_height, int samples_per_pixel, int max_depth, const hittable& world, std::vector<point_light> lights, point3 center, double pixel_samples_scale, point3 pixel00_loc, vec3 pixel_delta_u, vec3 pixel_delta_v, double defocus_angle, vec3 defocus_disk_u, vec3 defocus_disk_v) {
        // Get the pixel position from the block and thread indices
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        // Check that the pixel is within the image bounds
        if (i < image_width && j < image_height) {
            color pixel_color(0, 0, 0);

            for (int sample = 0; sample < samples_per_pixel; ++sample) {
                // Get the ray for the current pixel
                ray r = get_ray(i, j, pixel00_loc, pixel_delta_u, pixel_delta_v);
                pixel_color += ray_color(r, max_depth, world, lights);
            }

            // Apply the scaling factor for pixel samples
            pixel_color *= pixel_samples_scale;

            // Convert to gamma-corrected color values
            auto r = linear_to_gamma(pixel_color.x());
            auto g = linear_to_gamma(pixel_color.y());
            auto b = linear_to_gamma(pixel_color.z());

            // Clamp and convert color to 8-bit values
            static const interval intensity(0.000, 0.999);
            int rByte = static_cast<int>(255.999 * intensity.clamp(r));
            int gByte = static_cast<int>(255.999 * intensity.clamp(g));
            int bByte = static_cast<int>(255.999 * intensity.clamp(b));

            // Write the color to the pixel buffer
            int pixel_index = (j * image_width + i) * CHANNEL_NUM;
            d_pixels[pixel_index] = rByte;
            d_pixels[pixel_index + 1] = gByte;
            d_pixels[pixel_index + 2] = bByte;
        }
    }
};




#endif // CAMERA_H
