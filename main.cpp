#include <iostream>
#include <string>
#include <future>
#include <algorithm>
#include <fstream>
#include "vec3.h"
#include "ray.h"


#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


bool fileExists(const char* fileName) {
    std::ifstream infile(fileName);
    return infile.good();
}

using color = vec3; 


color ray_color(const ray& r) {
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5 * (unit_direction.y() + 1.0); // changes y dir to normalized length between 0 and 1
    return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0); // Blended value = (1-a) * startValue + a * endValue; 0 <= a <= 1
}

int main() {

    const int image_width = 1024;
    const auto aspect_ratio = 16.0 / 9.0;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // Viewport widths less than one are ok since they are real valued.
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(image_width) / image_height);

#define CHANNEL_NUM 3

    uint8_t* pixels = new uint8_t[image_width * image_height * CHANNEL_NUM];
    int index = 0;

    // Camera

    auto focal_length = 1.0;
    auto camera_center = point3(0, 0, 0);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    auto viewport_u = vec3(viewport_width, 0, 0); // x
    auto viewport_v = vec3(0, -viewport_height, 0); // y

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    auto pixel_delta_u = viewport_u / image_width;
    auto pixel_delta_v = viewport_v / image_height;

    // Calculate the location of the upper left pixel.
    auto viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
    auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);


    // Image

    

    const char* name = "Sky.png";

    if (fileExists(name)) {
        std::cout << "File already exists. " << name << std::endl;
        return 1;
    }

    // Render

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (float j = image_height - 1; j >= 0; --j) {
        float percent = 100 * (j / (image_height - 1));
        std::cerr << "\rPercent Rendered: " << static_cast <int>(100 - percent) << "% " << std::flush;
        for (int i = 0; i < image_width; ++i) {


            auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
            auto ray_direction = pixel_center - camera_center;
            ray r(camera_center, ray_direction);

            color pixel_color = ray_color(r);

            int ir = static_cast<int>(255.999 * pixel_color.x());
            int ig = static_cast<int>(255.999 * pixel_color.y());
            int ib = static_cast<int>(255.999 * pixel_color.z());

            pixels[index++] = ir;
            pixels[index++] = ig;
            pixels[index++] = ib;

        }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "\nDone rendering " << name << " in " << (std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()) / 1000 << " seconds" << std::endl;

    stbi_write_png(name, image_width, image_height, CHANNEL_NUM, pixels, image_width * CHANNEL_NUM);

    return 0;

}


