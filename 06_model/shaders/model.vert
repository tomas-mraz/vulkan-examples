#version 450

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inNormal;

layout (binding = 0) uniform UBO {
    mat4 mvp;
    mat4 model;
} ubo;

layout (location = 0) out vec3 outNormal;

void main() {
    gl_Position = ubo.mvp * vec4(inPosition, 1.0);
    outNormal = mat3(ubo.model) * inNormal;
}
