#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texturePos;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform uniforms {
    vec3 viewPos;
    vec3 ambientColor;
    vec3 diffuseColor;
    vec3 specularColor;
    vec3 specularExponent;
};

void main() {
    vec3 lightPos = vec3(0.0, 1.0, -3.0);

    float ambientStrength = 0.02;
    vec3 ambient = ambientStrength * ambientColor;

    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(lightPos - fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * diffuseColor;

    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 10);
    vec3 specular = 0.5 * spec * specularColor;

    outColor = vec4(diffuse + specular + ambient, 1.0);
}
