#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"
#include "texture.h"
#include "rtw_stb_image.h"

using color = vec3;

class material {
public:
	virtual ~material() = default;

	virtual color emitted(double u, double v, const point3& p) const {
		return color(0, 0, 0);
	}

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

class diffuse_light : public material {
public:
	diffuse_light(shared_ptr<texture> tex) : tex(tex) {}
	diffuse_light(const color& emit) : tex(make_shared<solid_color>(emit)) {}

	color emitted(double u, double v, const point3& p) const override {
		return tex->value(u, v, p);
	}
private:
	shared_ptr<texture> tex;
};
class emissive_light : public material {
public:
	emissive_light(shared_ptr<texture> tex) : tex(tex) {}
	emissive_light(const color& emit) : tex(make_shared<solid_color>(emit)) {}

	// Override the emitted function to keep the light emission behavior
	color emitted(double u, double v, const point3& p) const override {
		return tex->value(u, v, p);
	}

	// Override the scatter function to make sure the material doesn't scatter rays or contribute to the rendering
	bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override {
		return false; // No scattering, as we don't want the object to render itself
	}

private:
	shared_ptr<texture> tex;
};

class isotropic : public material {
public:
	isotropic(const color& albedo) : tex(make_shared<solid_color>(albedo)) {}
	isotropic(shared_ptr<texture> tex) : tex(tex) {}

	bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
		const override {
		scattered = ray(rec.p, random_unit_vector(), r_in.time());
		attenuation = tex->value(rec.u, rec.v, rec.p);
		return true;
	}

private:
	shared_ptr<texture> tex;
};

class specular : public material {
public:
	specular(const color& albedo, double shininess)
		: albedo(albedo), shininess(shininess) {}

	bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override {
		// Compute the reflected direction based on the incident ray and surface normal
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);

		// Introduce a small amount of diffuse scattering to simulate roughness
		vec3 diffuse = random_on_hemisphere(rec.normal);

		// Choose reflection or diffuse based on shininess (higher shininess = more reflective)
		double reflection_factor = std::pow(1.0 - dot(reflected, unit_vector(r_in.direction())), shininess);

		// Blend between diffuse and reflection based on the shininess
		vec3 scatter_direction = reflection_factor * reflected + (1.0 - reflection_factor) * diffuse;

		// Make sure the scattered direction is not too small
		if (scatter_direction.near_zero()) {
			scatter_direction = rec.normal;
		}

		// Create the scattered ray and attenuation
		scattered = ray(rec.p, scatter_direction, r_in.time());
		attenuation = albedo; // The material's albedo determines the intensity of the reflected light
		return true;
	}

private:
	color albedo;      // The color of the material
	double shininess;  // A factor that controls how specular the surface is (higher = shinier)
};


#endif 