#version 450

layout (location = 0) in vec3 inPosition;

layout (push_constant) uniform PushConstants {
    float offsetX;
    float offsetY;
    float angle;
    float aspect;
    float colorR;
    float colorG;
    float colorB;
    float brightness;
} pc;

void main() {
    // Rotate in square space, then apply aspect correction
    float c = cos(pc.angle);
    float s = sin(pc.angle);
    mat2 rot = mat2(c, s, -s, c);
    vec2 rotated = rot * inPosition.xy;
    gl_Position = vec4(rotated.x * pc.aspect + pc.offsetX, rotated.y + pc.offsetY, inPosition.z, 1.0);
}
