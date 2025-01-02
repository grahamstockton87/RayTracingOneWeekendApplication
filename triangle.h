//#pragma once
//
//#include "vec3.h"
//#include "hittable_list.h"
//#include "hittable.h"
//#include "ray.h"
//#include "material.h"
//#include <iostream>
//
//constexpr double kEpsilon = 1e-8;
//
//class triangle : public hittable {
//
//public:
//
//	triangle(vec3 point1, vec3 point2, vec3 point3, shared_ptr<material> m) : v0(point1), v1(point2), v2(point3), mat_ptr(m) {};
//
//	virtual bool hit(const ray& r, hit_record& rec)const;
//
//public:
//
//	vec3 v0;
//	vec3 v1;
//	vec3 v2;
//	shared_ptr<material> mat_ptr;
//
//};
//
//bool triangle::hit(const ray& r, hit_record& rec) const
//{
//
//	vec3 v0v1 = v1 - v0;
//	vec3 v0v2 = v2 - v0;
//
//	v0v1 = v2 - v0;
//	v0v2 = v1 - v0;
//
//	vec3 pvec = cross(r.dir, v0v2);
//	float det = dot(v0v1, pvec);
//	// if the determinant is negative the triangle is backfacing
//	// if the determinant is close to 0, the ray misses the triangle
//	if (det > -kEpsilon && det < kEpsilon) { return false; }
//
//	// ray and triangle are parallel if det is close to 0
//
//	if (fabs(det) < 0) { return false; }
//
//	float invDet = 1 / det;
//
//	vec3 tvec = r.orig - v0;
//
//	auto u = dot(tvec, pvec) * invDet;
//	if (u < 0 || u > 1) { return false; }
//
//	vec3 qvec = cross(tvec, v0v1);
//	auto v = dot(r.dir, qvec) * invDet;
//
//	if (v < 0 || u + v > 1) { return false; }
//	auto t = dot(v0v2, qvec) * invDet;
//
//	rec.mat_ptr = mat_ptr;
//	rec.normal = normalize(cross(v0v1, v0v2));
//	rec.p = r.at(t);
//	return true;
//}
//
//
