#ifndef HITTABLE_H
#define HITTABLE_H


class hit_record {
public:
	point3 p;
	vec3 normal;
	double t;
	bool front_face;

	// Sets normal on face to be outside or inside face with ray dir. Outward_normal (unit length)
	void set_face_normal(const ray& r, const vec3& outward_normal) {
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class hittable {
public:
	virtual ~hittable() = default;

	virtual bool hit(const ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const = 0;
};

#endif // !HITTABLE_H

