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
#include "constant_medium.h"

#include "tiny_obj_loader.h"


#include <string>
#include <future>
#include <algorithm>
#include <fstream>
#include <sstream>

#include "windows.h"


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


bool parseFaceVertex(const std::string & facePart, int& vertexIndex) {
    std::istringstream ss(facePart);
    std::string vertexIndexStr;

    // Read the vertex index (everything until the first '/')
    if (!std::getline(ss, vertexIndexStr, '/')) return false;

    vertexIndex = std::stoi(vertexIndexStr);
    return true;
}

bool loadObj(const std::string path, hittable_list& world, const shared_ptr<lambertian> mat) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return false;
    }

    std::vector<vec3> vertices;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string prefix;
        ss >> prefix;

        if (prefix == "v") {
            vec3 vertex;
            ss >> vertex[0] >> vertex[1] >> vertex[2];
            vertices.push_back(vertex);
            std::cout << "Vertex " << vertices.size() << ": "
                << vertex[0] << " " << vertex[1] << " " << vertex[2] << std::endl;
        }
        else if (prefix == "f") {
            std::vector<int> vertexIndices;

            std::string group;
            while (ss >> group) {
                std::istringstream groupStream(group);
                int v, vt, vn;
                char discard;

                groupStream >> v >> discard >> vt >> discard >> vn;
                vertexIndices.push_back(v);
            }

            std::cout << "Parsed face with vertices:";
            for (int idx : vertexIndices) {
                std::cout << " " << idx;
            }
            std::cout << std::endl;

            if (vertexIndices.size() < 3) {
                std::cerr << "Skipping malformed face with only " << vertexIndices.size() << " vertices." << std::endl;
                continue;
            }

            int maxIndex = static_cast<int>(vertices.size());
            bool valid = true;
            for (int idx : vertexIndices) {
                if (idx < 1 || idx > maxIndex) {
                    std::cerr << "Face references out-of-range vertex: " << idx
                        << " (only have " << maxIndex << " vertices)" << std::endl;
                    valid = false;
                    break;
                }
            }
            if (!valid) continue;

            if (vertexIndices.size() == 3) {
                world.add(make_shared<triangle>(
                    vertices[vertexIndices[0] - 1],
                    vertices[vertexIndices[1] - 1],
                    vertices[vertexIndices[2] - 1],
                    mat
                ));
            }
            else if (vertexIndices.size() == 4) {
                world.add(make_shared<triangle>(
                    vertices[vertexIndices[0] - 1],
                    vertices[vertexIndices[1] - 1],
                    vertices[vertexIndices[2] - 1],
                    mat
                ));
                world.add(make_shared<triangle>(
                    vertices[vertexIndices[0] - 1],
                    vertices[vertexIndices[2] - 1],
                    vertices[vertexIndices[3] - 1],
                    mat
                ));
            }
            else {
                std::cerr << "Skipping unsupported face with " << vertexIndices.size() << " vertices." << std::endl;
            }
        }
    }

    file.close();
    return true;
}
void LoadObjLib(const std::string& filename, hittable_list& world, const shared_ptr<lambertian> mat) {
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./"; // Path to look for MTL files (if any)

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filename, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error() << std::endl;
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader Warning: " << reader.Warning() << std::endl;
    }

    const tinyobj::attrib_t& attrib = reader.GetAttrib();
    const std::vector<tinyobj::shape_t>& shapes = reader.GetShapes();
    const std::vector<tinyobj::material_t>& materials = reader.GetMaterials();
    std::vector<vec3> vertices;

    // Loop over shapes (each shape is typically a mesh or object in the OBJ file)
    for (size_t s = 0; s < shapes.size(); s++) {
        //std::cout << "Shape " << s << ": " << shapes[s].name << std::endl;

        // Loop over faces (each face can have 3 or more vertices)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];

            // Loop over vertices in the face
            for (size_t v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                // Access vertex data
                float vx = attrib.vertices[3 * idx.vertex_index + 0];
                float vy = attrib.vertices[3 * idx.vertex_index + 1];
                float vz = attrib.vertices[3 * idx.vertex_index + 2];

                vec3 vert(vx, vy, vz);
                vertices.push_back(vert);
                //std::cout << "  v(" << vx << ", " << vy << ", " << vz << ")";

                if (idx.normal_index >= 0) {
                    float nx = attrib.normals[3 * idx.normal_index + 0];
                    float ny = attrib.normals[3 * idx.normal_index + 1];
                    float nz = attrib.normals[3 * idx.normal_index + 2];
                    //std::cout << " n(" << nx << ", " << ny << ", " << nz << ")";
                }

                if (idx.texcoord_index >= 0) {
                    float tx = attrib.texcoords[2 * idx.texcoord_index + 0];
                    float ty = attrib.texcoords[2 * idx.texcoord_index + 1];
                    //std::cout << " t(" << tx << ", " << ty << ")";
                }

                std::cout << std::endl;
            }
            index_offset += fv;
        }
    }
    //  make triangles
    for (int i = 0; i < vertices.size();i+=3) {
        world.add(make_shared<triangle>(vertices[i], vertices[i+1], vertices[i+2], mat));
    }
}




int main() {



    const char* image_name = "TriangleBoxes.png";

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
    case 4:
    {
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

    }case 6: {
        
        
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

        }case 7: {


        auto red = make_shared<lambertian>(color(.65, .05, .05));
        auto green = make_shared<lambertian>(color(.12, .45, .15));
        auto light = make_shared<diffuse_light>(color(50, 50, 50));


        //world.add(make_shared<triangle>(point3(0, 0, 0), vec3(0, 10, 0), vec3(10, 0, 0),  red));

        //world.add(make_shared<triangle>(point3(10, 0, 0), vec3(10, 10, 0), vec3(0, 10, 0),red));
        //shared_ptr<hittable> quad1 = triangle_quad(point3(0, 0, 0), 10, 10, red);


        //world.add(quad1);
        //if (loadObj("C:/Users/graha/Documents/Visual Studio Projects 2024/Coding Projects/RayTracingOneWeekendApplication/RayTracingOneWeekendApplication/monkey.obj", world, red)) {
        //    std::cout << "Sucessful Loading Object";
        //};
        LoadObjLib("C:/Users/graha/Documents/Visual Studio Projects 2024/Coding Projects/RayTracingOneWeekendApplication/RayTracingOneWeekendApplication/car.obj", world, red);
        world.add(make_shared<sphere>(point3(0, 0, 15), 4, light));

        cam.samples_per_pixel = 10;
        cam.max_depth = 50;
        cam.background = color(0, 0, 0);

        cam.vfov = 20;
        cam.lookfrom = point3(8, 5, 10);
        cam.lookat = point3(0, 0, -5);
        cam.vup = vec3(0, 1, 0);

        cam.defocus_angle = 0;
        break;
    }
    }


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


