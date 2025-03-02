#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "vec3.h"
#include "hittable.h"
#include "interval.h"
#include "aabb.h"
#include <memory>  // Include memory for shared_ptr

constexpr double kEpsilon = 1e-8;

class triangle : public hittable {
public:
    // Stationary Triangle Constructor
    triangle(vec3 p0, vec3 p1, vec3 p2, std::shared_ptr<material> mat)
        : p0(p0), p1(p1), p2(p2), mat(mat) {
        // Compute AABB for a stationary triangle

        vec3 n = cross(p1 - p0, p2 - p0); // Compute normal using cross product
        normal = unit_vector(n);           // Normalize the normal
        D = dot(normal, p0);               // Plane equation constant

        set_bounding_box();
    }
    virtual void set_bounding_box() {
        // Compute the bounding box of all four vertices.
        auto bbox_diagonal1 = aabb(p0, p0 + p1 + p2);
        auto bbox_diagonal2 = aabb(p0 + p1, p0 + p2);
        bbox = aabb(bbox_diagonal1, bbox_diagonal2);
    }


    aabb bounding_box() const override { return bbox; }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        // Triangle edges
        vec3 v0v1 = p1 - p0;
        vec3 v0v2 = p2 - p0;

        // Determinant for Möller-Trumbore intersection test
        vec3 pvec = cross(r.direction(), v0v2);
        float det = dot(v0v1, pvec);

        // Ray parallel to triangle
        if (fabs(det) < kEpsilon) return false;

        float invDet = 1.0f / det;
        vec3 tvec = r.origin() - p0;

        // Compute barycentric coordinate u
        auto u = dot(tvec, pvec) * invDet;
        if (u < 0.0f || u > 1.0f) return false;

        // Compute barycentric coordinate v
        vec3 qvec = cross(tvec, v0v1);
        auto v = dot(r.direction(), qvec) * invDet;
        if (v < 0.0f || u + v > 1.0f) return false;

        // Compute t (intersection distance)
        auto t = dot(v0v2, qvec) * invDet;
        if (t < ray_t.min || t > ray_t.max) return false;

        point3 intersection = r.at(t);

        // Compute barycentric coordinates for UV interpolation
        float alpha = (1 - u - v);
        float beta = u;
        float gamma = v;

        if (!is_interior(alpha, beta, rec)) return false;

        // Interpolate UV coordinates based on barycentric coordinates
        rec.u = alpha * uv0.x() + beta * uv1.x() + gamma * uv2.x();
        rec.v = alpha * uv0.y() + beta * uv1.y() + gamma * uv2.y();

        // Set hit record
        rec.t = t;
        rec.p = intersection;
        rec.mat = mat;
        rec.set_face_normal(r, normal);

        return true;
    }

    virtual bool is_interior(double a, double b, hit_record& rec) const {
        interval unit_interval = interval(0, 1);
        // Check if the barycentric coordinates are within the triangle
        if (!unit_interval.contains(a) || !unit_interval.contains(b))
            return false;

    }

private:
    vec3 p0, p1, p2;            // Triangle vertices
    std::shared_ptr<material> mat;
    aabb bbox;

    // Interpolates the triangle's position at a given time t (unused if stationary)
    void get_triangle_at_time(double time, vec3& v0, vec3& v1, vec3& v2) const {
        v0 = p0;
        v1 = p1;
        v2 = p2;
    }

private:
    vec3 uv0 = vec3(0, 0, 0);
    vec3 uv1 = vec3(1, 0, 0);
    vec3 uv2 = vec3(0, 1, 0);

    vec3 normal;   // Normal of the triangle plane
    double D;      // Plane equation constant
};


inline shared_ptr<hittable_list> triangle_quad(const point3& orig, double height, double width, shared_ptr<material> mat)
{
    // Returns the 2d quad

    auto sides = make_shared<hittable_list>();
    // Left triangle (counter-clockwise)
    sides->add(make_shared<triangle>(
        point3(orig),
        vec3(orig.x(), height + orig.x(), orig.z()),
        vec3(width + orig.x(), orig.y(), orig.z()),
        mat
    ));

    // Right triangle (counter-clockwise)
    sides->add(make_shared<triangle>(
        point3(orig.x() + width, orig.y(), orig.z()),
        vec3(orig.x() + width, orig.y() + height, orig.z()),
        vec3(orig.x(), height + orig.y(), orig.z()),
        mat
    ));


    return sides;
}

#endif