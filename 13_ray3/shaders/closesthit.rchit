/* Copyright (c) 2023, Sascha Willems
 *
 * SPDX-License-Identifier: MIT
 *
 */

#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(location = 0) rayPayloadInEXT vec4 hitValue;
layout(location = 2) rayPayloadEXT bool shadowed;
hitAttributeEXT vec2 attribs;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 2, set = 0) uniform CameraProperties
{
	mat4 viewInverse;
	mat4 projInverse;
	uint frame;
	vec4 lightPos;
} cam;
layout(binding = 3, set = 0) uniform sampler2D image;

struct GeometryNode {
	uint64_t vertexBufferDeviceAddress;
	uint64_t indexBufferDeviceAddress;
	int textureIndexBaseColor;
	int textureIndexOcclusion;
};
layout(binding = 4, set = 0) buffer GeometryNodes { GeometryNode nodes[]; } geometryNodes;

layout(binding = 5, set = 0) uniform sampler2D textures[];

#include "bufferreferences.glsl"
#include "geometrytypes.glsl"

void main()
{
	Triangle tri = unpackTriangle(gl_PrimitiveID);
	hitValue = vec4(tri.normal, gl_HitTEXT);

	GeometryNode geometryNode = geometryNodes.nodes[gl_GeometryIndexEXT];

	vec3 color = texture(textures[nonuniformEXT(geometryNode.textureIndexBaseColor)], tri.uv).rgb;
	if (geometryNode.textureIndexOcclusion > -1) {
		float occlusion = texture(textures[nonuniformEXT(geometryNode.textureIndexOcclusion)], tri.uv).r;
		color *= occlusion;
	}

	// Transform normal from object space to world space
	vec3 worldNormal = normalize(mat3(gl_ObjectToWorldEXT) * tri.normal);

	// Compute hit position and lighting
	vec3 hitPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
	vec3 toLight = cam.lightPos.xyz - hitPos;
	float dist = length(toLight);
	vec3 L = normalize(toLight);
	float NdotL = max(dot(worldNormal, L), 0.0);

	// Red point light with attenuation
	float attenuation = 1.0 / (1.0 + 2.0 * dist * dist);
	vec3 lightColor = vec3(1.0, 1.0, 1.0) * NdotL * attenuation * 5.0;
	float ambient = 0.08;
	vec3 litColor = color * (vec3(ambient) + lightColor);

	// Shadow casting
	float tmin = 0.001;
	float epsilon = 0.001;
	vec3 origin = hitPos + worldNormal * epsilon;
	shadowed = true;
	traceRayEXT(topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, 0, 0, 1, origin, tmin, L, dist, 2);
	if (shadowed) {
		litColor = color * vec3(ambient);
	}
	hitValue = vec4(litColor, gl_HitTEXT);
}
