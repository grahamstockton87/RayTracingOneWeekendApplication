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

	double vfov = 90; // field of view
	point3 lookfrom = point3(0, 0, 0); // point cam looking from
	point3 lookat = point3(0, 0, -1); // point cam looking at 
	vec3 vup = vec3(0, 1, 0); // relative up direction 

	void render(const hittable& world) {
		initialize();

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		for (int j = 0; j < image_height; ++j) {
			float percent = 100.0f * (1.0f - (j / float(image_height - 1)));
			std::cerr << "\rPercent Rendered: " << static_cast <int>(100 - percent) << "% " << std::flush;
			for (int i = 0; i < image_width; ++i) {
				color pixel_color(0, 0, 0);

				for (int sample = 0; sample < samples_per_pixel; sample++) {
					ray r = get_ray(i, j);
					pixel_color += ray_color(r, max_depth, world);
				}
				pixel_color *= pixel_samples_scale;
				// clamp rgb values [0,1} to byte range [0,255]
				

				auto r = linear_to_gamma(pixel_color.x());
				auto g = linear_to_gamma(pixel_color.y());
				auto b = linear_to_gamma(pixel_color.z());

				static const interval intensity(0.000, 0.999);
				int rByte = static_cast<int>(255.999 * intensity.clamp(r));
				int gByte = static_cast<int>(255.999 * intensity.clamp(g));
				int bByte = static_cast<int>(255.999 * intensity.clamp(b));

				pixels[index++] = rByte;
				pixels[index++] = gByte;
				pixels[index++] = bByte;

			}
		}

		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

		double time_taken = (std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
		std::cout << "\nDone rendering " << image_name << " in " << time_taken/60 << " seconds" << std::endl;

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

	uint8_t* pixels;
	int index;

	void initialize() {
		image_height = int(image_width / aspect_ratio);
		image_height = (image_height < 1) ? 1 : image_height;

		pixel_samples_scale = 1.0 / samples_per_pixel;

		center = lookfrom;

		// Determine viewport dimensions.
		auto focal_length = (lookfrom - lookat).length();
		auto theta = degrees_to_radians(vfov);
		auto h = std::tan(theta / 2);
		auto viewport_height = 2 * h * focal_length;
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
		auto viewport_upper_left = center - (focal_length * w) - viewport_u / 2 - viewport_v / 2;
		pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

		pixels = new uint8_t[image_width * image_height * CHANNEL_NUM];

	}

	ray get_ray(int i, int j) const {
		// create ray from camera origin directed randomly sampled
		//point is i j

		auto offset = sample_square();
		auto pixel_sample = pixel00_loc
				+ ((i + offset.x()) * pixel_delta_u)
				+ ((j + offset.y()) * pixel_delta_v);
		auto ray_origin = center;
		auto ray_direction = pixel_sample - ray_origin;

		return ray(ray_origin, ray_direction);
	}
	vec3 sample_square() const {
		//return vector between point in [-0.5,-0.5] - [0.5, 0.5] in the unit square;
		return vec3(random_double() - 0.5, random_double() - 0.5, 0);
	}
	// grabs color of ray if it intersects
	color ray_color(const ray& r, int depth, const hittable& world) const {
		// limit child rays 
		if (depth <= 0)
			return color(0, 0, 0);

		hit_record rec;

		if (world.hit(r, interval(0.001, infinity), rec)) {
			//vec3 direction = random_on_hemisphere(rec.normal); 
			ray scatttered;
			color attenuation;
			if (rec.mat->scatter(r, rec, attenuation, scatttered)) {
				//return color(0.0, 1.0, 0.0);
				return attenuation * ray_color(scatttered, depth - 1, world);
			}
			return color(0.0, 0.0, 0.0);
		}

		vec3 unit_direction = unit_vector(r.direction());
		auto a = 0.5 * (unit_direction.y() + 1.0);
		return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
		//return color(0.0, 1.0, 0.0);
	}
};



#endif // CAMERA_H