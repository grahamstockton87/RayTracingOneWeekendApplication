#ifndef POINT_LIGHT
#define POINT_LIGHT

#include "vec3.h"
#include <vector>

using color = vec3;

class point_light {
public:
    point_light(point3 position, color intensity, double size = 1.0)
        : position(position), intensity(intensity), size(size) {}

    point3 get_position() const { return position; }
    color get_intensity() const { return intensity; }
    double get_size() const { return size; }  // Get size of the point light

private:
    point3 position;
    color intensity;
    double size;  // The size of the point light
};

#endif // !POINT_LIGHT
