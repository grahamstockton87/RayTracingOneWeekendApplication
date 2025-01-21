#pragma once

#include "vec3.h"
#include "hittable.h"
#include "interval.h"
#include <memory>  // Include memory for shared_ptr

constexpr double kEpsilon = 1e-8;

class triangle : public hittable {

public:
    triangle(vec3 point1, vec3 point2, vec3 point3, std::shared_ptr<material> mat)
        : v0(point1), v1(point2), v2(point3), mat(mat) {}

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {

        // Edges of the triangle
        vec3 v0v1 = v1 - v0;
        vec3 v0v2 = v2 - v0;

        // Calculate the determinant (denominator of the barycentric coordinates)
        vec3 pvec = cross(r.direction(), v0v2);
        float det = dot(v0v1, pvec);

        // If the determinant is close to 0, the ray is parallel to the triangle
        if (det > -kEpsilon && det < kEpsilon) return false;

        // Calculate the inverse determinant
        float invDet = 1.0f / det;

        // Vector from the first vertex to the ray's origin
        vec3 tvec = r.origin() - v0;

        // Calculate the u parameter (barycentric coordinate)
        auto u = dot(tvec, pvec) * invDet;
        if (u < 0.0f || u > 1.0f) return false;

        // Calculate the v parameter (barycentric coordinate)
        vec3 qvec = cross(tvec, v0v1);
        auto v = dot(r.direction(), qvec) * invDet;
        if (v < 0.0f || u + v > 1.0f) return false;

        // Calculate the t parameter (ray intersection with triangle)
        auto t = dot(v0v2, qvec) * invDet;

        // If the intersection occurs outside the given ray interval, return false
        if (t < ray_t.min || t > ray_t.max) return false;

        // Record the hit details
        rec.mat = mat;  // Use mat_ptr here to assign the material
        rec.normal = unit_vector(cross(v0v1, v0v2));  // The normal is perpendicular to the triangle
        rec.p = r.at(t);
        return true;
    }

private:
    vec3 v0, v1, v2;                   // Vertices of the triangle
    std::shared_ptr<material> mat;  // Material for the triangle
};
