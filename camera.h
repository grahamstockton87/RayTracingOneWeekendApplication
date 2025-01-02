#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "vec3.h"
#include "ray.h"
#include "interval.h"

#include <string>
#include <future>
#include <algorithm>
#include <fstream>

#include "windows.h"
#include <iostream>

#define CHANNEL_NUM 3

#include "stb_image_write.h"

using color = vec3;

class camera {
public:

	const int image_width = 1024;

	const double aspect_ratio = 16.0 / 9.0;

	const char* image_name = "New Camera WHODIS";

	void rename_image(const char* image_name_in) {
		image_name = image_name_in;
	}

	void render(const hittable& world) {
		initialize();

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		for (int j = 0; j < image_height; ++j) {
			float percent = 100.0f * (1.0f - (j / float(image_height - 1)));
			std::cerr << "\rPercent Rendered: " << static_cast <int>(100 - percent) << "% " << std::flush;
			for (int i = 0; i < image_width; ++i) {


				auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
				auto ray_direction = pixel_center - center;
				ray r(center, ray_direction);

				color pixel_color = ray_color(r, world);

				int ir = static_cast<int>(255.999 * pixel_color.x());
				int ig = static_cast<int>(255.999 * pixel_color.y());
				int ib = static_cast<int>(255.999 * pixel_color.z());

				pixels[index++] = ir;
				pixels[index++] = ig;
				pixels[index++] = ib;

			}
		}

		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

		double time_taken = (std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
		std::cout << "\nDone rendering " << image_name << " in " << time_taken << " miliseconds" << std::endl;

		stbi_write_png(image_name, image_width, image_height, CHANNEL_NUM, pixels, image_width * CHANNEL_NUM); 
	}

private:

	int    image_height;   // Rendered image height
	point3 center;         // Camera center
	point3 pixel00_loc;    // Location of pixel 0, 0
	vec3   pixel_delta_u;  // Offset to pixel to the right
	vec3   pixel_delta_v;  // Offset to pixel below

	uint8_t* pixels;
	int index;

	void initialize() {
		image_height = int(image_width / aspect_ratio);
		image_height = (image_height < 1) ? 1 : image_height;

		center = point3(0, 0, 0);

		// Determine viewport dimensions.
		auto focal_length = 1.0;
		auto viewport_height = 2.0;
		auto viewport_width = viewport_height * (double(image_width) / image_height);

		// Calculate the vectors across the horizontal and down the vertical viewport edges.
		auto viewport_u = vec3(viewport_width, 0, 0);
		auto viewport_v = vec3(0, -viewport_height, 0);

		// Calculate the horizontal and vertical delta vectors from pixel to pixel.
		pixel_delta_u = viewport_u / image_width;
		pixel_delta_v = viewport_v / image_height;

		// Calculate the location of the upper left pixel.
		auto viewport_upper_left =
			center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
		pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

		pixels = new uint8_t[image_width * image_height * CHANNEL_NUM];

	}
	// grabs color of ray if it intersects
	color ray_color(const ray& r, const hittable& world) const {
		hit_record rec;

		if (world.hit(r, interval(0, infinity), rec)) {
			return 0.5 * (rec.normal + color(1, 1, 1));
		}

		vec3 unit_direction = unit_vector(r.direction());
		auto a = 0.5 * (unit_direction.y() + 1.0);
		return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
	}
};



#endif // CAMERA_H