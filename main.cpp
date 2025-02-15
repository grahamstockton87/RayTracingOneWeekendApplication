#include "rtweekend.h"

#include "bvh.h"
#include "hittable_list.h"
#include "hittable.h"
#include "sphere.h"
#include "triangle.h"
#include "camera.h"
#include "quad.h"
#include "point_light.h"
//#include "cameraGPU.cuh"
#include "texture.h"


#include <string>
#include <future>
#include <algorithm>
#include <fstream>

#include "windows.h"


#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "main.h"

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



    const char* image_name = "PointLight.png";

    int scene = 4;

    // World
    hittable_list world;
    camera cam;
    // A container for all lights in the scene
    std::vector<point_light> lights;

    switch (scene) {
    case 0: {
        auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
        auto checker = make_shared<checker_texture>(0.32, color(0, 0, 0), color(.9, .9, .9));
        world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, make_shared<lambertian>(checker)));


        auto material1 = make_shared<dielectric>(1.5);
        world.add(make_shared<sphere>(point3(2, 1, 5), 1.0, material1));

        auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
        auto pertext = make_shared<noise_texture>(10);
        world.add(make_shared<sphere>(point3(-2, 1, 5), 1.0, make_shared<lambertian>(pertext)));

        auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
        //world.add(make_shared<sphere>(point3(0, 1, 5), 1.0, material3));

        auto material4 = make_shared<metal>(color(1.0, 1.0, 1.0), 0.0);

        auto checkerT = make_shared<checker_texture_triangle>(0.5, color(0, 0, 0), color(.9, .9, .9));
        world.add(make_shared<triangle>(
            point3(4, 0, 8), point3(-4, 0, 8), point3(0, 6, 8),
            (make_shared<lambertian>(checkerT))));

        auto earth_texture = make_shared<image_texture>("C:/Users/graha/Documents/Visual Studio Projects 2024/Coding Projects/RayTracingOneWeekendApplication/RayTracingOneWeekendApplication/earthmap.jpg");
        auto earth_surface = make_shared<lambertian>(earth_texture);
        auto globe = make_shared<sphere>(point3(0, 1, 5), 1.0, earth_surface);
        world.add(globe);

        cam.image_name = image_name;
        cam.samples_per_pixel = 10;
        cam.max_depth = 50;

        cam.vfov = 20;
        cam.lookfrom = point3(1, 4, -10);
        cam.lookat = point3(0, 1, 5);
        cam.vup = vec3(0, 1, 0);

        cam.defocus_angle = 0.1;
        cam.focus_dist = point_distance(cam.lookat, cam.lookfrom);
        //cam.render(world, lights);

        break;
    }
    case 1: {
        // Materials
        auto left_red = make_shared<lambertian>(color(1.0, 0.2, 0.2));
        auto back_green = make_shared<lambertian>(color(0.2, 1.0, 0.2));
        auto right_blue = make_shared<lambertian>(color(0.2, 0.2, 1.0));
        auto upper_orange = make_shared<lambertian>(color(1.0, 0.5, 0.0));
        auto lower_teal = make_shared<lambertian>(color(0.2, 0.8, 0.8));

        // Quads
        world.add(make_shared<quad>(point3(-3, -2, 5), vec3(0, 0, -4), vec3(0, 4, 0), left_red));
        world.add(make_shared<quad>(point3(-2, -2, 0), vec3(4, 0, 0), vec3(0, 4, 0), back_green));
        world.add(make_shared<quad>(point3(3, -2, 1), vec3(0, 0, 4), vec3(0, 4, 0), right_blue));
        world.add(make_shared<quad>(point3(-2, 3, 1), vec3(4, 0, 0), vec3(0, 0, 4), upper_orange));
        world.add(make_shared<quad>(point3(-2, -3, 5), vec3(4, 0, 0), vec3(0, 0, -4), lower_teal));

        break;
    }
    case 2: {
        auto pertext = make_shared<noise_texture>(4);
        world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, make_shared<lambertian>(pertext)));
        world.add(make_shared<sphere>(point3(0, 2, 0), 2, make_shared<lambertian>(pertext)));

        auto difflight = make_shared<diffuse_light>(color(10, 10, 10));
        world.add(make_shared<sphere>(point3(0, 7, 0), 2, difflight));
        world.add(make_shared<quad>(point3(3, 1, -2), vec3(2, 0, 0), vec3(0, 2, 0), difflight));

        cam.samples_per_pixel = 1000;
        cam.max_depth = 50;
        cam.background = color(0, 0, 0);

        cam.vfov = 20;
        cam.lookfrom = point3(26, 3, 6);
        cam.lookat = point3(0, 2, 0);
        cam.vup = vec3(0, 1, 0);

        cam.defocus_angle = 0;
    }
    case 3: {
        auto red = make_shared<lambertian>(color(.65, .05, .05));
        auto white = make_shared<lambertian>(color(.73, .73, .73));
        auto green = make_shared<lambertian>(color(.12, .45, .15));
        auto light = make_shared<diffuse_light>(color(15, 15, 15));

        world.add(make_shared<quad>(point3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), green));
        world.add(make_shared<quad>(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), red));
        world.add(make_shared<quad>(point3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105), light));
        world.add(make_shared<quad>(point3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555), white));
        world.add(make_shared<quad>(point3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), white));
        world.add(make_shared<quad>(point3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), white));

        world.add(box(point3(130, 0, 65), point3(295, 165, 230), white));
        world.add(box(point3(265, 0, 295), point3(430, 330, 460), white));


        cam.samples_per_pixel = 200;
        cam.max_depth = 5;
        cam.background = color(0, 0, 0);

        cam.vfov = 40;
        cam.lookfrom = point3(278, 278, -800);
        cam.lookat = point3(278, 278, 0);
        cam.vup = vec3(0, 1, 0);

        cam.defocus_angle = 0;
    }
    case 4:
    {
        auto red = make_shared<lambertian>(color(.65, .05, .05));

        world.add(make_shared<sphere>(point3(0, 2, 4), 1.0, red));

        point_light light(point3(0, 2, 2), color(1, 1, 1), 0.1);
        lights.push_back(light);

        auto difflight = make_shared<diffuse_light>(color(10, 10, 10));
        world.add(make_shared<sphere>(point3(0, 7, 0), 2, difflight));

        cam.samples_per_pixel = 200;
        cam.max_depth = 5;
        cam.background = color(0, 0, 0);

        cam.vfov = 40;
        cam.lookfrom = point3(0, 0, 0);
        cam.lookat = point3(0, 2, 4);
        cam.vup = vec3(0, 1, 0);
    }
    }





    //for (int a = -2; a < 3; a++) {
    //    for (int b = -2; b < 3; b++) {
    //        auto choose_mat = random_double();
    //        point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

    //        if ((center - point3(4, 0.2, 0)).length() > 0.9) {
    //            shared_ptr<material> sphere_material;

    //            if (choose_mat < 0.8) {
    //                // diffuse
    //                auto albedo = color::random() * color::random();
    //                sphere_material = make_shared<lambertian>(albedo);
    //                auto center2 = center + vec3(0, random_double(0, .5), 0);
    //                world.add(make_shared<sphere>(center, center2, 0.2, sphere_material));
    //            }
    //            else if (choose_mat < 0.95) {
    //                // metal
    //                auto albedo = color::random(0.5, 1);
    //                auto fuzz = random_double(0, 0.5);
    //                sphere_material = make_shared<metal>(albedo, fuzz);
    //                world.add(make_shared<sphere>(center, 0.2, sphere_material));
    //            }
    //            else {
    //                // glass
    //                sphere_material = make_shared<dielectric>(1.5);
    //                world.add(make_shared<sphere>(center, 0.2, sphere_material));
    //            }
    //        }
    //    }
    //}


    world = hittable_list(make_shared<bvh_node>(world));

//  CAMERA SETTINGS ---------------------------------------------------------------------------

    cam.image_name = image_name;


// --------------------------------------------------------------------------------------------

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
    cam.render(world, lights);


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


