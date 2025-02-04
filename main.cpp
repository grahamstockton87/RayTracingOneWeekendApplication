#include "rtweekend.h"

#include "hittable_list.h"
#include "hittable.h"
#include "sphere.h"
#include "triangle.h"
#include "camera.h"

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

// gives distance from point to point 
inline double point_distance(const point3 point, const point3 lookfrom) {
    return (std::sqrt(
        pow(point.x() - lookfrom.x(), 2) +
        pow(point.y() - lookfrom.y(), 2) +
        pow(point.z() - lookfrom.z(), 2)
    ));
}


int main() {

    // World

    hittable_list world;
    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -2; a < 3; a++) {
        for (int b = -2; b < 3; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    auto center2 = center + vec3(0, random_double(0, .5), 0);
                    world.add(make_shared<sphere>(center, center2, 0.2, sphere_material));
                }
                else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(2, 1, 5), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-2, 1, 5), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(0, 1, 5), 1.0, material3));

    auto material4 = make_shared<metal>(color(1.0, 1.0, 1.0), 0.0);
    world.add(make_shared<triangle>(
        point3(4, 0, 8), point3(-4, 0, 8), point3(0, 6, 8), 
        point3(4, 1, 8), point3(-4, 1, 8), point3(0, 7, 8),
        material3));

    const char* image_name = "AABB.png";

    camera cam;

    cam.image_name = image_name;
    cam.samples_per_pixel = 50;
    cam.max_depth = 50;

    cam.vfov = 20;
    cam.lookfrom = point3(1, 4, -10);
    cam.lookat = point3(0, 1, 5);
    cam.vup = vec3(0, 1, 0);

    cam.defocus_angle = 0.1;
    cam.focus_dist = point_distance(cam.lookat, cam.lookfrom);

   // cam.setName(image_name);

    // Image

    if (fileExists(cam.image_name)) {
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
    cam.render(world);


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


