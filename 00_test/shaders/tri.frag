#version 450

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

layout (location = 0) out vec4 outColor;

void main() {
    outColor = vec4(pc.colorR * pc.brightness, pc.colorG * pc.brightness, pc.colorB * pc.brightness, 1.0);
}
