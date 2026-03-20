#version 460
#extension GL_EXT_ray_tracing : enable

layout(location = 0) rayPayloadInEXT vec3 hitValue;

void main()
{
	// Sky gradient
	hitValue = vec3(0.4, 0.6, 1.0);
}
