#version 450

layout (binding = 1) uniform sampler2D texSampler;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inTexCoord;

layout (location = 0) out vec4 outColor;

void main() {
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    vec3 normal = normalize(inNormal);
    float diff = max(dot(normal, lightDir), 0.0);
    float ambient = 0.2;
    float lighting = ambient + diff * 0.8;

    vec4 texColor = texture(texSampler, inTexCoord);
    outColor = vec4(texColor.rgb * lighting, texColor.a);
}
