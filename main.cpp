#include <iostream>
#include <string>
#include <future>
#include <algorithm>
#include <fstream>


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

const int image_width = 1024;
const auto aspect_ratio = 16.0 / 9.0;
const int image_height = static_cast<int>(image_width / aspect_ratio);

#define CHANNEL_NUM 3

uint8_t* pixels = new uint8_t[image_width * image_height * CHANNEL_NUM];
int index = 0;

int main() {

    // Image

    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 400;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = static_cast<int>(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    const char* name = "ImageOutputColors.png";

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

            double r = double(i) / (image_width - 1);
            double g = double(j) / (image_height - 1);
            double b = 0.0;

            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);

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


