#include "rtweekend.h"

#include "hittable_list.h"
#include "hittable.h"
#include "sphere.h"

#include <string>
#include <future>
#include <algorithm>
#include <fstream>

#include "windows.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Function to convert const char* to std::wstring
std::wstring ConvertToWideString(const std::string& narrowString) {
    int wideSize = MultiByteToWideChar(CP_UTF8, 0, narrowString.c_str(), -1, nullptr, 0);
    std::wstring wideString(wideSize, 0);
    MultiByteToWideChar(CP_UTF8, 0, narrowString.c_str(), -1, &wideString[0], wideSize);
    return wideString;
}

bool fileExists(const char* fileName) {
    std::ifstream infile(fileName);
    return infile.good();
}

bool delete_image(const char* filename) {
    if (std::remove(filename) != 0) {
        perror("Error deleting file");
        return false;
    }
    else {
        std::cout << "File successfully deleted" << std::endl;
        return true;
    }
}

using color = vec3; 

inline double clamp(double value, double min_val, double max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

color ray_color(const ray& r, const hittable& world) {
    hit_record rec;

    if (world.hit(r, interval(0, infinity), rec)) {
        //return 0.5 * (rec.normal + color(1, 1, 1));
        double intensity = dot(r.direction(),rec.normal);
        double inverted_intensity = 256 - intensity;
        return color(inverted_intensity, inverted_intensity, inverted_intensity);
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}



int main() {


    const int image_width = 1024;
    const auto aspect_ratio = 16.0 / 9.0;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // STB Image Pixels

#define CHANNEL_NUM 3

    uint8_t* pixels = new uint8_t[image_width * image_height * CHANNEL_NUM];
    int index = 0;

    // World

    hittable_list world;

    world.add(make_shared<sphere>(point3(0, 0, -1), 0.5));
    world.add(make_shared<sphere>(point3(0, -100.5, -1), 100));


    // Viewport widths less than one are ok since they are real valued.
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(image_width) / image_height);


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

    
    const char* image_name = "TESTING.png";

    if (fileExists(image_name)) {
        std::cout << "File already exists. Would you like to delete? (y/n)" << std::endl;
        std::string input;
        std::cin >> input;
        if (input == "y") {
            if (!delete_image(image_name)) { return -1; }
        }
        else {
            return 0;
        }
        
    }


    // Render

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (int j = 0; j < image_height; ++j) {
        float percent = 100.0f * (1.0f - (j / float(image_height - 1)));
        std::cerr << "\rPercent Rendered: " << static_cast <int>(100 - percent) << "% " << std::flush;
        for (int i = 0; i < image_width; ++i) {


            auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
            auto ray_direction = pixel_center - camera_center;
            ray r(camera_center, ray_direction);

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

    // Convert file name to wide string
    std::wstring image_name_wide_string = ConvertToWideString(image_name);

    // Use ShellExecuteW to open the file with the default viewer
    HINSTANCE result = ShellExecuteW(NULL, L"open", image_name_wide_string.c_str(), NULL, NULL, SW_SHOWNORMAL);

    // Check if the operation was successful
    if ((INT_PTR)result <= 32) {
        // If the result is less than or equal to 32, it indicates an error
        MessageBox(NULL, L"Failed to open the image file.", L"Error", MB_OK | MB_ICONERROR);
    }

    return 0;

}


