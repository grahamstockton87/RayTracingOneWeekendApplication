#include "rtweekend.h"

#include "bvh.h"
#include "hittable_list.h"
#include "hittable.h"
#include "sphere.h"
#include "triangle.h"
#include "camera.h"
#include "quad.h"
#include "point_light.h"
#include "texture.h"
#include "constant_medium.h"
#include "mesh.h"

#include <string>
#include <future>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <regex>

#include "windows.h"


#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "mesh.h"

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


std::string getNextFilename(const std::string& originalName) {

    // Break name into "base" and "extension"
    size_t lastDot = originalName.find_last_of('.');
    std::string base = (lastDot == std::string::npos) ? originalName : originalName.substr(0, lastDot);
    std::string extension = (lastDot == std::string::npos) ? "" : originalName.substr(lastDot);

    // Regex to match (number) at the end
    std::regex numberedRegex(R"(^(.*)\((\d+)\)$)");
    std::smatch match;

    int number = 1;

    if (std::regex_match(base, match, numberedRegex)) {
        // Found something like "image(3)"
        base = match[1].str();      // the part before (number)
        number = std::stoi(match[2].str()) + 1; // Increment number
    }
    else {
        base = base; // no (n), start at (1)
    }

    // Search for the first unused filename
    std::string newName;

    newName = base + "(" + std::to_string(number) + ")" + extension;
    number++;

    return newName;
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

bool parseFaceVertex(const std::string & facePart, int& vertexIndex) {
    std::istringstream ss(facePart);
    std::string vertexIndexStr;

    // Read the vertex index (everything until the first '/')
    if (!std::getline(ss, vertexIndexStr, '/')) return false;

    vertexIndex = std::stoi(vertexIndexStr);
    return true;
}

int main() {



    std::string image_name_working = "Specular.png";

    int scene = 7;

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

        shared_ptr<hittable> box1 = box(point3(0, 0, 0), point3(165, 330, 165), white);
        box1 = make_shared<rotate_y>(box1, 15);
        box1 = make_shared<translate>(box1, vec3(265, 0, 295));
        world.add(box1);

        shared_ptr<hittable> box2 = box(point3(0, 0, 0), point3(165, 165, 165), white);
        box2 = make_shared<rotate_y>(box2, -18);
        box2 = make_shared<translate>(box2, vec3(130, 0, 65));
        world.add(box2);


        cam.samples_per_pixel = 5000;
        cam.max_depth = 25;
        cam.background = color(0, 0, 0);

        cam.vfov = 40;
        cam.lookfrom = point3(278, 278, -800);
        cam.lookat = point3(278, 278, 0);
        cam.vup = vec3(0, 1, 0);

        cam.defocus_angle = 0;
        break;
    }
    case 4: {
        auto red = make_shared<lambertian>(color(.65, .05, .05));

        world.add(make_shared<sphere>(point3(0, 2, 4), 1.0, red));

        //point_light light(point3(0, 2, 2), color(1, 1, 1), 0.1);
        //lights.push_back(light);

        // Create an emissive material and assign it to an object
        shared_ptr<material> emissive = make_shared<emissive_light>(color(1.0, 1.0, 1.0)); // White light

        auto difflight = make_shared<emissive_light>(color(10, 10, 10));
        world.add(make_shared<sphere>(point3(0, 4, 0), 3, emissive));

        cam.samples_per_pixel = 200;
        cam.max_depth = 5;
        cam.background = color(0, 0, 0);

        cam.vfov = 40;
        cam.lookfrom = point3(0, 0, 0);
        cam.lookat = point3(0, 2, 4);
        cam.vup = vec3(0, 1, 0);
        break;
    }
    case 5: {
        
        hittable_list boxes1;
        auto ground = make_shared<lambertian>(color(0.48, 0.83, 0.53));

        int boxes_per_side = 20;
        for (int i = 0; i < boxes_per_side; i++) {
            for (int j = 0; j < boxes_per_side; j++) {
                auto w = 100.0;
                auto x0 = -1000.0 + i * w;
                auto z0 = -1000.0 + j * w;
                auto y0 = 0.0;
                auto x1 = x0 + w;
                auto y1 = random_double(1, 101);
                auto z1 = z0 + w;

                boxes1.add(box(point3(x0, y0, z0), point3(x1, y1, z1), ground));
            }
        }

        hittable_list world;

        world.add(make_shared<bvh_node>(boxes1));

        auto light = make_shared<diffuse_light>(color(7, 7, 7));
        world.add(make_shared<quad>(point3(123, 554, 147), vec3(300, 0, 0), vec3(0, 0, 265), light));

        auto center1 = point3(400, 400, 200);
        auto center2 = center1 + vec3(30, 0, 0);
        auto sphere_material = make_shared<lambertian>(color(0.7, 0.3, 0.1));
        world.add(make_shared<sphere>(center1, center2, 50, sphere_material));

        world.add(make_shared<sphere>(point3(260, 150, 45), 50, make_shared<dielectric>(1.5)));
        world.add(make_shared<sphere>(
            point3(0, 150, 145), 50, make_shared<metal>(color(0.8, 0.8, 0.9), 1.0)
        ));

        auto boundary = make_shared<sphere>(point3(360, 150, 145), 70, make_shared<dielectric>(1.5));
        world.add(boundary);
        world.add(make_shared<constant_medium>(boundary, 0.2, color(0.2, 0.4, 0.9)));
        boundary = make_shared<sphere>(point3(0, 0, 0), 5000, make_shared<dielectric>(1.5));
        world.add(make_shared<constant_medium>(boundary, .0001, color(1, 1, 1)));

        auto emat = make_shared<lambertian>(make_shared<image_texture>("earthmap.jpg"));
        world.add(make_shared<sphere>(point3(400, 200, 400), 100, emat));
        auto pertext = make_shared<noise_texture>(0.2);
        world.add(make_shared<sphere>(point3(220, 280, 300), 80, make_shared<lambertian>(pertext)));

        hittable_list boxes2;
        auto white = make_shared<lambertian>(color(.73, .73, .73));
        int ns = 1000;
        for (int j = 0; j < ns; j++) {
            boxes2.add(make_shared<sphere>(point3::random(0, 165), 10, white));
        }

        world.add(make_shared<translate>(
            make_shared<rotate_y>(
                make_shared<bvh_node>(boxes2), 15),
            vec3(-100, 270, 395)
        )
        );

        cam.background = color(0, 0, 0);
        cam.samples_per_pixel = 10;
        cam.vfov = 40;
        cam.lookfrom = point3(478, 278, -600);
        cam.lookat = point3(278, 278, 0);
        cam.vup = vec3(0, 1, 0);

        cam.defocus_angle = 0;
        break;

    }
    case 6: {
        
        
        auto red = make_shared<lambertian>(color(.65, .05, .05));
        auto white = make_shared<lambertian>(color(.73, .73, .73));
        auto green = make_shared<lambertian>(color(.12, .45, .15));
        auto light = make_shared<diffuse_light>(color(7, 7, 7));


        world.add(make_shared<quad>(point3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), green));
        world.add(make_shared<quad>(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), red));
        world.add(make_shared<quad>(point3(113, 554, 127), vec3(330, 0, 0), vec3(0, 0, 305), light));
        world.add(make_shared<quad>(point3(0, 555, 0), vec3(555, 0, 0), vec3(0, 0, 555), white));
        world.add(make_shared<quad>(point3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555), white));
        world.add(make_shared<quad>(point3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), white));

        shared_ptr<hittable> box1 = box(point3(0, 0, 0), point3(165, 330, 165), white);
        box1 = make_shared<rotate_y>(box1, 15);
        box1 = make_shared<translate>(box1, vec3(265, 0, 295));

        shared_ptr<hittable> box2 = box(point3(0, 0, 0), point3(165, 165, 165), white);
        box2 = make_shared<rotate_y>(box2, -18);
        box2 = make_shared<translate>(box2, vec3(130, 0, 65));

        world.add(make_shared<constant_medium>(box1, 0.005, color(0, 0, 0)));
        world.add(make_shared<constant_medium>(box2, 0.005, color(0.2, 0.2, 0.2)));

        cam.samples_per_pixel = 2000;
        cam.max_depth = 10;
        cam.background = color(0, 0, 0);

        cam.vfov = 40;
        cam.lookfrom = point3(278, 278, -800);
        cam.lookat = point3(278, 278, 0);
        cam.vup = vec3(0, 1, 0);

        cam.defocus_angle = 0;
        break;

        }
    case 7: {

        auto male_texture = make_shared<image_texture>("C:/Users/graha/Documents/Visual Studio Projects 2024/Coding Projects/RayTracingOneWeekendApplication/RayTracingOneWeekendApplication/male_texture.jpg");
        auto male_material = make_shared<lambertian>(male_texture);
        auto red = make_shared<lambertian>(color(.65, .05, .05));
        auto grey = make_shared<lambertian>(color(0.1,0.1,0.1));
        auto green = make_shared<lambertian>(color(.12, .45, .15));
        auto light = make_shared<diffuse_light>(color(20, 20, 20));
        auto checkerT = make_shared<checker_texture_triangle>(0.5, color(0, 0, 0), color(.9, .9, .9));
        auto noiseT = make_shared<noise_texture>(20);
        auto noise = make_shared<lambertian>(noiseT);
        auto checker = make_shared<lambertian>(checkerT);
        auto metalMat = make_shared<metal>(color(0,0,0),0.2);

        auto corgi_texture = make_shared<image_texture>("corgi_diffuse.jpeg");
        auto corgi_material = make_shared<lambertian>(corgi_texture);


        //mesh car;
        //car.rotate(90, glm::vec3(0, 1, 0));
        // === Apply a 90-degree rotation on Y-axis ===
        //glm::mat4 transform = glm::mat4(1.0f);
        //transform = glm::translate(transform, glm::vec3(3, 0, 0));
        //transform = glm::rotate(transform, glm::radians(90.0f), glm::vec3(0, 1, 0));
        //glm::vec3 scaleFactor(1.5f, 1.5f, 1.5f);
        //transform = glm::scale(transform, scaleFactor);

        // === Load the object and apply the transform ===
        //if (!car.loadObj("corgi.obj", world, corgi_material, transform)) {
        //    std::cerr << "Failed to load OBJ." << std::endl;
        //    return -1;
        //}
  
        //world.add(make_shared<quad>(point3(0, 6, 0), vec3(30, 6, 0), vec3(30, 6, 30), light));
        //make_shared<translate>(base, vec3(0, 0, 0));
        world.add(make_shared<sphere>(point3(0, -1005, 0), 1000, grey));

        world.add(make_shared<sphere>(point3(0, 15, 0), 5, light));

        auto specular_material = make_shared<specular>(color(1.0,0.1,0.1), 5);
        world.add(make_shared<sphere>(point3(-5, 0, 0), 5, specular_material));

        auto metal_material = make_shared<metal>(color(1.0, 0.1, 0.1), 0.0);
        //world.add(make_shared<sphere>(point3(5, 0, 0), 5, metal_material));

        cam.samples_per_pixel = 100;
        cam.max_depth = 10;
        cam.background = color(0, 0, 0);

        cam.vfov = 90;
        cam.lookfrom = point3(0, 5, -10);
        cam.lookat = point3(0,0,0);
        cam.vup = vec3(0, 1, 0);
        cam.focus_dist = point_distance(cam.lookat, cam.lookfrom) - 2.5;

        cam.defocus_angle = 0;
        break;
    }
    }


    world = hittable_list(make_shared<bvh_node>(world));

//  FILE SETTINGS ---------------------------------------------------------------------------

    if (fileExists(cam.image_name)) {
        getNextFilename(image_name_working);
    }
    cam.image_name = image_name_working.c_str();


// --------------------------------------------------------------------------------------------

    // Image

    //if (fileExists(cam.image_name)) {
        //std::cout << "File already exists. Would you like to delete? (y/n)" << std::endl;
        //std::string input;
        //std::cin >> input;
        //if (input == "y") {
        //    if (!delete_image(image_name)) { return -1; }
        //}
        //else {
        //    return 0;
        //}
        
    //}

    std::cout << "Rendering Image: " << cam.image_name << std::endl;
    // Render
    cam.render(world, lights);

    // Convert file name to wide string
    std::wstring image_name_wide_string = ConvertToWideString(cam.image_name);

    // Use ShellExecuteW to open the file with the default viewer
    HINSTANCE result = ShellExecuteW(NULL, L"open", image_name_wide_string.c_str(), NULL, NULL, SW_SHOWNORMAL);

    // Check if the operation was successful
    if ((INT_PTR)result <= 32) {
        // If the result is less than or equal to 32, it indicates an error
        MessageBox(NULL, L"Failed to open the image file.", L"Error", MB_OK | MB_ICONERROR);
    }

    return 0;

}


