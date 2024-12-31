#ifndef SPHERE_H
#define SPHERE_H

#include "vec3.h"
#include "hittable.h"
#include "interval.h"


class sphere : public hittable {
public:
	sphere(const point3& center, double radius) : center(center), radius(std::fmax(0,radius)) {}

	bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
		vec3 oc = center - r.origin(); // normal vector
		double a = r.direction().length_squared();
		double h = dot(r.direction(), oc);
		double c = oc.length_squared() - radius * radius;

		double discriminant = h * h - a * c;
		if (discriminant < 0)
			return false; // no hits
		double sqrtd = std::sqrt(discriminant);

		double root = (h - sqrtd) / a;
		if(!ray_t.surrounds(root)){
			root = (h + sqrtd) / a;
			if (!ray_t.surrounds(root))
				return false;
		}
		rec.t = root;
		rec.p = r.at(rec.t);
		vec3 outward_normal = (rec.p - center) / radius;
		rec.set_face_normal(r, outward_normal); // make normal outwawrd facing

		return true;
	}
private:
	point3 center;
	double radius;
};

#endif // SPHERE_H