#version 450

layout (location = 0) in vec3 inNormal;
layout (location = 0) out vec4 outColor;

void main() {
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    vec3 normal = normalize(inNormal);
    float diff = max(dot(normal, lightDir), 0.0);
    float ambient = 0.15;
    float lighting = ambient + diff * 0.85;
    vec3 baseColor = vec3(0.4, 0.5, 0.7);
    outColor = vec4(baseColor * lighting, 1.0);
}
