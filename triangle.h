#pragma once

#include "vec3.h"
#include "hittable.h"
#include "interval.h"
#include "aabb.h"
#include <memory>  // Include memory for shared_ptr

constexpr double kEpsilon = 1e-8;

class triangle : public hittable {
public:
    // Stationary Triangle Constructor
    triangle(vec3 point1, vec3 point2, vec3 point3, std::shared_ptr<material> mat)
        : v0_start(point1), v1_start(point2), v2_start(point3), mat(mat)
    {
        // Compute AABB for a stationary triangle
        bbox = compute_aabb(v0_start, v1_start, v2_start);
        std::cout << "default constructor";
    }

    // Moving Triangle Constructor
    triangle(const vec3& p0_start, const vec3& p1_start, const vec3& p2_start,
        const vec3& p0_end, const vec3& p1_end, const vec3& p2_end,
        shared_ptr<material> mat)
        : v0_start(p0_start), v1_start(p1_start), v2_start(p2_start),
        v0_end(p0_end), v1_end(p1_end), v2_end(p2_end), mat(mat)
    {
        // Compute bounding boxes at t0 and t1
        aabb box1 = compute_aabb(v0_start, v1_start, v2_start);
        aabb box2 = compute_aabb(v0_end, v1_end, v2_end);

        // Merge AABBs to get the full bounding range
        bbox = aabb(box1, box2);
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        //// Interpolate the triangle position at ray time
        //vec3 v0, v1, v2;
        //get_triangle_at_time(r.time(), v0, v1, v2);

        //// Triangle edges
        //vec3 v0v1 = v1 - v0;
        //vec3 v0v2 = v2 - v0;

        //// Determinant for Möller-Trumbore intersection test
        //vec3 pvec = cross(r.direction(), v0v2);
        //float det = dot(v0v1, pvec);

        //// Ray parallel to triangle
        //if (fabs(det) < kEpsilon) return false;

        //float invDet = 1.0f / det;
        //vec3 tvec = r.origin() - v0;

        //// Compute barycentric coordinate u
        //auto u = dot(tvec, pvec) * invDet;
        //if (u < 0.0f || u > 1.0f) return false;

        //// Compute barycentric coordinate v
        //vec3 qvec = cross(tvec, v0v1);
        //auto v = dot(r.direction(), qvec) * invDet;
        //if (v < 0.0f || u + v > 1.0f) return false;

        //// Compute t (intersection distance)
        //auto t = dot(v0v2, qvec) * invDet;
        //if (t < ray_t.min || t > ray_t.max) return false;

        //// Record the hit
        //rec.mat = mat;
        //rec.p = r.at(t);
        //rec.normal = unit_vector(cross(v0v1, v0v2));
        //rec.t = t;
        //rec.u = (1 - u - v) * uv0.x() + u * uv1.x() + v * uv2.x();
        //rec.v = (1 - u - v) * uv0.y() + u * uv1.y() + v * uv2.y();
        //return true;
            // compute plane's normal
            vec3 v0, v1, v2;
            get_triangle_at_time(r.time(), v0, v1, v2);

            vec3 v0v1 = v1 - v0;
            vec3 v0v2 = v2 - v0;

            // Compute normal using cross product
            vec3 N = cross(v0v1, v0v2);

            // Check if the ray is parallel to the triangle
            float NdotRayDirection = dot(N, r.direction());
            if (fabs(NdotRayDirection) < kEpsilon) return false;

            // Compute intersection distance t
            float t = (dot(N, v0) - dot(N, r.origin())) / NdotRayDirection;

            // If t is outside the valid range, return false
            if (t < ray_t.min || t > ray_t.max) return false;

            // Compute intersection point
            vec3 P = r.at(t);

            // Compute the total area of the triangle
            float area = N.length() / 2;

            // Compute barycentric coordinates
            vec3 C;

            // Compute u (for triangle BCP)
            vec3 v1p = P - v1;
            vec3 v1v2 = v2 - v1;
            C = cross(v1v2, v1p);
            float u = (C.length() / 2) / area;
            if (dot(N, C) < 0) return false;

            // Compute v (for triangle CAP)
            vec3 v2p = P - v2;
            vec3 v2v0 = v0 - v2;
            C = cross(v2v0, v2p);
            float v = (C.length() / 2) / area;
            if (dot(N, C) < 0) return false;

            // Compute w (for triangle ABP)
            vec3 v0p = P - v0;
            C = cross(v0v1, v0p);
            if (dot(N, C) < 0) return false;

            // Compute real UV coordinates using barycentric interpolation
            float w = 1.0f - u - v;
            rec.u = (1 - u - v) * uv0.x() + u * uv1.x() + v * uv2.x();
            rec.v = (1 - u - v) * uv0.y() + u * uv1.y() + v * uv2.y();

            // The point is inside the triangle
            rec.t = t;
            rec.p = P;
            rec.normal = unit_vector(N);
            rec.mat = mat;

            return true;
        }


    aabb bounding_box() const override { return bbox; }

private:
    vec3 v0_start, v1_start, v2_start; // Triangle start position
    vec3 v0_end, v1_end, v2_end;       // Triangle end position
    std::shared_ptr<material> mat;
    aabb bbox;

    // Interpolates the triangle's position at a given time t
    void get_triangle_at_time(double time, vec3& v0, vec3& v1, vec3& v2) const {
        v0 = v0_start + time * (v0_end - v0_start);
        v1 = v1_start + time * (v1_end - v1_start);
        v2 = v2_start + time * (v2_end - v2_start);
    }
    // Computes AABB for given triangle vertices
    aabb compute_aabb(const vec3& p0, const vec3& p1, const vec3& p2) const {
        vec3 min_point(
            fmin(p0.x(), fmin(p1.x(), p2.x())),
            fmin(p0.y(), fmin(p1.y(), p2.y())),
            fmin(p0.z(), fmin(p1.z(), p2.z()))
        );

        vec3 max_point(
            fmax(p0.x(), fmax(p1.x(), p2.x())),
            fmax(p0.y(), fmax(p1.y(), p2.y())),
            fmax(p0.z(), fmax(p1.z(), p2.z()))
        );

        return aabb(min_point, max_point);
    }
private:
    vec3 uv0 = vec3(0, 0, 0);
    vec3 uv1 = vec3(1, 0, 0);
    vec3 uv2 = vec3(0, 1, 0);
};

// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates.html