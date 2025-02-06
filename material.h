#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"
#include "texture.h"
#include "rtw_stb_image.h"

using color = vec3;

class material {
public:
	virtual ~material() = default;

	virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const {
		return false;
	}
};
class lambertian : public material {
public:

	lambertian(const color& albedo) : tex(make_shared<solid_color>(albedo)) {}
	lambertian(shared_ptr<texture> tex) : tex(tex) {}


	bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override {
		vec3 scatter_direction = rec.normal + random_unit_vector();

		if (scatter_direction.near_zero()) {
			scatter_direction = rec.normal;
		}
		scattered = ray(rec.p, scatter_direction, r_in.time());
		attenuation = tex->value(rec.u,rec.v,rec.p);
		return true;
	}
private:
	shared_ptr<texture> tex;
};

class dielectric : public material {
public:
	dielectric(double refraction_index) : refraction_index(refraction_index) {}

	bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override {
		attenuation = color(1.0, 1.0, 1.0);
		double ri = rec.front_face ? (1.0 / refraction_index) : refraction_index;

		vec3 unit_direction = unit_vector(r_in.direction());
		double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
		double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

		bool cannot_refract = ri * sin_theta > 1.0;
		vec3 direction;

		if (cannot_refract || reflectance(cos_theta, ri) > random_double())
			direction = reflect(unit_direction, rec.normal);
		else
			direction = refract(unit_direction, rec.normal, ri);

		scattered = ray(rec.p, direction, r_in.time());
		return true;
	}
private:
	double refraction_index;

	static double reflectance(double cosine, double refraction_index) {
		// Use Schlick's approximation for reflectance.
		auto r0 = (1 - refraction_index) / (1 + refraction_index);
		r0 = r0 * r0;
		return r0 + (1 - r0) * std::pow((1 - cosine), 5);
	}

};

class metal : public material {
public:
	metal(const color& albedo, double fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

	bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override {
		vec3 reflected = reflect(r_in.direction(), rec.normal);
		reflected = unit_vector(reflected) + (fuzz * random_unit_vector());
		scattered = scattered = ray(rec.p, reflected, r_in.time());
		attenuation = albedo;
		return (dot(scattered.direction(), rec.normal) > 0);
	}
private:
	color albedo;
	double fuzz;
};

class image_texture : public texture {
public:
	image_texture(const char* filename) : image(filename) {}

	color value(double u, double v, const point3& p) const override {
		// If we have no texture data, then return solid cyan as a debugging aid.
		if (image.width() <= 0 || image.height() <= 0) return color(0, 1, 1);

		// Clamp input texture coordinates to [0,1] x [1,0]
		u = interval(0, 1).clamp(u);
		v = 1.0 - interval(0, 1).clamp(v);  // Flip V to image coordinates

		auto i = int(u * image.width());
		auto j = int(v * image.height());
		auto pixel = image.pixel_data(i, j);

		auto color_scale = 1.0 / 255.0;
		return color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
	}

private:
	rtw_image image;
};


#endif 