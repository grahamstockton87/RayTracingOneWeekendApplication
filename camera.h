#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "vec3.h"
#include "ray.h"
#include "interval.h"
#include "material.h"

#include <string>
#include <future>
#include <algorithm>
#include <fstream>

#include "windows.h"
#include <iostream>

#define CHANNEL_NUM 3

#include "stb_image_write.h"

using color = vec3;

// Conver linear to gamma
inline double linear_to_gamma(double linear_component) {
	if (linear_component > 0) {
		return std::sqrt(linear_component);
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
	color background = vec3(0,0,0); // scene background color

	double vfov = 90; // field of view
	point3 lookfrom = point3(0, 0, 0); // point cam looking from
	point3 lookat = point3(0, 0, -1); // point cam looking at 
	vec3 vup = vec3(0, 1, 0); // relative up direction

	double defocus_angle = 0;  // Variation angle of rays through each pixel
	double focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus

	void render(const hittable& world) {
		initialize();

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		const int num_threads = std::thread::hardware_concurrency(); // Number of threads to use
		std::vector<std::future<void>> futures; // Store futures for async tasks
		const int rows_per_thread = image_height / num_threads;

		std::atomic<int> rows_completed = 0; // Atomic counter for progress tracking

		auto render_rows = [&](int start_row, int end_row) {
			for (int j = start_row; j < end_row; ++j) {
				for (int i = 0; i < image_width; ++i) {
					color pixel_color(0, 0, 0);

					for (int sample = 0; sample < samples_per_pixel; ++sample) {
						ray r = get_ray(i, j);
						pixel_color += ray_color(r, max_depth, world);
					}
					pixel_color *= pixel_samples_scale;

					// Clamp and gamma-correct pixel colors
					auto r = linear_to_gamma(pixel_color.x());
					auto g = linear_to_gamma(pixel_color.y());
					auto b = linear_to_gamma(pixel_color.z());

					static const interval intensity(0.000, 0.999);
					int rByte = static_cast<int>(255.999 * intensity.clamp(r));
					int gByte = static_cast<int>(255.999 * intensity.clamp(g));
					int bByte = static_cast<int>(255.999 * intensity.clamp(b));

					int pixel_index = (j * image_width + i) * CHANNEL_NUM;
					pixels[pixel_index] = rByte;
					pixels[pixel_index + 1] = gByte;
					pixels[pixel_index + 2] = bByte;
				}
				rows_completed++;
			}
			};

		// Divide work among threads
		for (int t = 0; t < num_threads; ++t) {
			int start_row = t * rows_per_thread;
			int end_row = (t == num_threads - 1) ? image_height : start_row + rows_per_thread;
			futures.push_back(std::async(std::launch::async, render_rows, start_row, end_row));
		}
		// Progress monitoring loop
		while (rows_completed < image_height) {
			float percent = 100.0f * rows_completed / image_height;
			std::cerr << "\rPercent Rendered: " << static_cast<int>(percent) << "% " << std::flush;
			std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Update every 100ms
		}
		// Wait for all threads to complete
		for (auto& f : futures) {
			f.get();
		}


		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

		double time_taken = (std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
		std::cout << "\nDone rendering " << image_name << " in " << time_taken / 60 << " seconds" << std::endl;

		stbi_write_png(image_name, image_width, image_height, CHANNEL_NUM, pixels, image_width * CHANNEL_NUM);
	}

private:

	int    image_height;   // Rendered image height
	double pixel_samples_scale; // color scale factor for sum of pixels
	point3 center;         // Camera center
	point3 pixel00_loc;    // Location of pixel 0, 0
	vec3   pixel_delta_u;  // Offset to pixel to the right
	vec3   pixel_delta_v;  // Offset to pixel below
	vec3 u, v, w; // cam frame basis vectors
	vec3   defocus_disk_u;       // Defocus disk horizontal radius
	vec3   defocus_disk_v;       // Defocus disk vertical radius

	uint8_t* pixels;
	int index;

	void initialize() {
		image_height = int(image_width / aspect_ratio);
		image_height = (image_height < 1) ? 1 : image_height;

		pixel_samples_scale = 1.0 / samples_per_pixel;

		center = lookfrom;

		// Determine viewport dimensions.

		auto theta = degrees_to_radians(vfov);
		auto h = std::tan(theta / 2);
		auto viewport_height = 2 * h * focus_dist;
		auto viewport_width = viewport_height * (double(image_width) / image_height);

		// calc u v w basis unit vectors for cord frame
		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);

		// Calculate the vectors across the horizontal and down the vertical viewport edges.
		auto viewport_u = viewport_width * u; // Vector across viewport horizontal edge
		auto viewport_v = viewport_height * -v; // Vector down viewport vertical edge

		// Calculate the horizontal and vertical delta vectors from pixel to pixel.
		pixel_delta_u = viewport_u / image_width;
		pixel_delta_v = viewport_v / image_height;

		// Calculate the location of the upper left pixel.
		auto viewport_upper_left = center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
		pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

		// Calculate the camera defocus disk basis vectors.
		auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
		defocus_disk_u = u * defocus_radius;
		defocus_disk_v = v * defocus_radius;

		pixels = new uint8_t[image_width * image_height * CHANNEL_NUM];

	}

	ray get_ray(int i, int j) const {
		// Construct a camera ray originating from the defocus disk and directed at a randomly
	   // sampled point around the pixel location i, j.

		auto offset = sample_square();
		auto pixel_sample = pixel00_loc
			+ ((i + offset.x()) * pixel_delta_u)
			+ ((j + offset.y()) * pixel_delta_v);
		auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
		auto ray_direction = pixel_sample - ray_origin;
		auto ray_time = random_double();

		return ray(ray_origin, ray_direction, ray_time);

	}
	vec3 sample_square() const {
		//return vector between point in [-0.5,-0.5] - [0.5, 0.5] in the unit square;
		return vec3(random_double() - 0.5, random_double() - 0.5, 0);
	}
	point3 defocus_disk_sample() const {
		// Returns a random point in the camera defocus disk.
		auto p = random_in_unit_disk();
		return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
	}

	// grabs color of ray if it intersects
	color ray_color(const ray& r, int depth, const hittable& world) const {
		// limit child rays 
		if (depth <= 0)
			return color(0, 0, 0);

		hit_record rec;

		if (!world.hit(r, interval(0.001, infinity), rec)) 
			return background;
 
		ray scattered;
		color attenuation;
		color color_from_emission = rec.mat->emitted(rec.u, rec.v, rec.p);

		if (!rec.mat->scatter(r, rec, attenuation, scattered)) 
			return color_from_emission;
		
		color color_from_scatter = attenuation * ray_color(scattered, depth - 1, world);

		return color_from_emission + color_from_scatter;
	}
};



#endif // CAMERA_H