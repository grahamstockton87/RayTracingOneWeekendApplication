#ifndef MESH_H
#define MESH_H

#include "vec3.h"
#include "hittable_list.h"
#include "triangle.h"
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

class mesh {
public:
    mesh() {}
    mesh(const std::vector<glm::mat4>& triangles)
        : mesh_matrices(triangles)
    {}

    /// Loads a .obj file and stores each triangle as a 4x4 matrix
    bool loadObj(const std::string path, hittable_list& world, const shared_ptr<lambertian> mat, glm::mat4 transform) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << path << std::endl;
            return false;
        }

        std::vector<glm::vec3> temp_vertices;
        std::vector<glm::vec2> temp_uvs;  // Store UV coordinates
        std::vector<int> vertex_indices;
        std::vector<int> uv_indices;  // Store UV indices

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream ss(line);
            std::string prefix;
            ss >> prefix;

            // === Parse vertices ===
            if (prefix == "v") {
                glm::vec3 vertex;
                ss >> vertex.x >> vertex.y >> vertex.z;
                temp_vertices.push_back(vertex);
            }

            // === Parse texture coordinates (UVs) ===
            else if (prefix == "vt") {
                glm::vec2 uv;
                ss >> uv.x >> uv.y;
                temp_uvs.push_back(uv);
            }

            // === Parse faces ===
            else if (prefix == "f") {
                std::vector<int> indices;
                std::vector<int> uv_indices_face;  // UV indices for this face
                std::string group;
                while (ss >> group) {
                    std::istringstream groupStream(group);
                    int v, vt, vn;
                    char discard;

                    // Handles format: "f v/vt/vn"
                    groupStream >> v >> discard >> vt >> discard >> vn;
                    indices.push_back(v - 1);  // Convert OBJ 1-based to 0-based
                    uv_indices_face.push_back(vt - 1);  // Convert OBJ 1-based to 0-based
                }

                if (indices.size() < 3) continue;

                // If face is a triangle
                if (indices.size() == 3) {
                    addTriangle(temp_vertices, temp_uvs, indices[0], indices[1], indices[2], uv_indices_face, mat, world, transform);
                }

                // If face is a quad, split into two triangles
                else if (indices.size() == 4) {
                    addTriangle(temp_vertices, temp_uvs, indices[0], indices[1], indices[2], uv_indices_face, mat, world, transform);
                    addTriangle(temp_vertices, temp_uvs, indices[0], indices[2], indices[3], uv_indices_face, mat, world, transform);
                }

                // Unsupported face size
                else {
                    std::cerr << "Skipping face with " << indices.size() << " vertices." << std::endl;
                }
            }
        }

        file.close();
        return true;
    }

    /// Adds a triangle as a 4x4 matrix (homogeneous coordinates) with UV coordinates
    void addTriangle(
        const std::vector<glm::vec3>& vertices,
        const std::vector<glm::vec2>& uvs,
        int v0, int v1, int v2,
        const std::vector<int>& uv_indices,
        const shared_ptr<lambertian> mat,
        hittable_list& world,
        const glm::mat4& transform
    ) {
        // === Convert each vertex to homogeneous coordinates ===
        glm::vec4 p0 = glm::vec4(vertices[v0], 1.0f);
        glm::vec4 p1 = glm::vec4(vertices[v1], 1.0f);
        glm::vec4 p2 = glm::vec4(vertices[v2], 1.0f);

        // === Apply the transformation matrix ===
        p0 = transform * p0;
        p1 = transform * p1;
        p2 = transform * p2;

        // === Store the triangle as a matrix in mesh_matrices ===
        glm::mat4 triangle_matrix(1.0f);
        triangle_matrix[0] = p0;
        triangle_matrix[1] = p1;
        triangle_matrix[2] = p2;
        triangle_matrix[3] = glm::vec4(0, 0, 0, 1); // Homogeneous base

        mesh_matrices.push_back(triangle_matrix);

        // === Convert back to vec3 for the triangle class ===
        vec3 pos0(p0.x, p0.y, p0.z);
        vec3 pos1(p1.x, p1.y, p1.z);
        vec3 pos2(p2.x, p2.y, p2.z);

        // === UV coordinates for the triangle ===
        glm::vec2 uv0 = uvs[uv_indices[0]];
        glm::vec2 uv1 = uvs[uv_indices[1]];
        glm::vec2 uv2 = uvs[uv_indices[2]];

        // === Add it to the world ===
        world.add(make_shared<triangle>(pos0, pos1, pos2, mat, uv0, uv1, uv2));
    }

    /// Apply a transformation matrix (scale, rotate, translate)
    void applyTransform(const glm::mat4& transform) {
        for (auto& matrix : mesh_matrices) {
            for (int i = 0; i < 3; ++i) {
                matrix[i] = transform * matrix[i];
            }
        }
    }

    /// Scale the mesh by a factor
    void scale(float factor) {
        glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(factor));
        applyTransform(scaleMatrix);
    }

    /// Rotate the mesh around an axis
    void rotate(float angle, const glm::vec3& axis) {
        glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(angle), axis);
        applyTransform(rotationMatrix);
    }

    /// Translate the mesh
    void translate(const glm::vec3& offset) {
        glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), offset);
        applyTransform(translationMatrix);
    }

    /// Upload the mesh to CUDA memory (optional)
    void uploadToCuda(uint8_t*& d_mesh) {
        cudaMalloc(&d_mesh, mesh_matrices.size() * sizeof(glm::mat4));
        cudaMemcpy(d_mesh, mesh_matrices.data(), mesh_matrices.size() * sizeof(glm::mat4), cudaMemcpyHostToDevice);
    }

public:
    std::vector<glm::mat4> mesh_matrices;
};

#endif
