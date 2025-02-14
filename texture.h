#ifndef TEXTURE_H
#define TEXTURE_H

#include "vec3.h"
#include <cmath>
#include <algorithm>
#include "rtweekend.h"
#include "rtw_stb_image.h"
#include "perlin.h"

using color = vec3;

class texture {
public:
    virtual ~texture() = default;

    virtual color value(double u, double v, const point3& p) const = 0;
};

class solid_color : public texture {
public:
    solid_color(const color& albedo) : albedo(albedo) {}

    solid_color(double red, double green, double blue) : solid_color(color(red, green, blue)) {}

    color value(double u, double v, const point3& p) const override {
        return albedo;
    }

private:
    color albedo;
};

class checker_texture : public texture {
public:
    checker_texture(double scale, shared_ptr<texture> even, shared_ptr<texture> odd)
        : inv_scale(1.0 / scale), even(even), odd(odd) {}

    checker_texture(double scale, const color& c1, const color& c2)
        : checker_texture(scale, make_shared<solid_color>(c1), make_shared<solid_color>(c2)) {}

    color value(double u, double v, const point3& p) const override {
        auto xInteger = int(std::floor(inv_scale * p.x()));
        auto yInteger = int(std::floor(inv_scale * p.y()));
        auto zInteger = int(std::floor(inv_scale * p.z()));

        bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;

        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }

private:
    double inv_scale;
    shared_ptr<texture> even;
    shared_ptr<texture> odd;
};

class checker_texture_triangle : public texture {
public:
    checker_texture_triangle(double scale, shared_ptr<texture> even, shared_ptr<texture> odd)
        : inv_scale(1.0 / std::max(0.01, scale)), even(even), odd(odd) {}

    checker_texture_triangle(double scale, const color& c1, const color& c2)
        : checker_texture_triangle(scale, make_shared<solid_color>(c1), make_shared<solid_color>(c2)) {}

    color value(double u, double v, const point3& p) const override {
        // Flip v for proper UV alignment
        v = 1.0 - v;

        // Scale UV coordinates properly
        auto uInteger = int(std::round(inv_scale * u * image_width));
        auto vInteger = int(std::round(inv_scale * v * image_height));

        bool isEven = (uInteger + vInteger) % 2 == 0;
        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }

private:
    double inv_scale;
    shared_ptr<texture> even;
    shared_ptr<texture> odd;
    const int image_width = 10;  // Adjust based on actual texture size
    const int image_height = 10;
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

class noise_texture : public texture {
public:
    noise_texture(double scale) : scale(scale) {}

    color value(double u, double v, const point3& p) const override {
        return color(1, 1, 1) * noise.noise(scale*p);
    }
private:
    perlin noise;
    double scale;
};

#endif