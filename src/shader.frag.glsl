#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texturePos;
layout(location = 0) out vec4 outColor;

struct Style
{
    vec3 ambient_color;
    vec3 diffuse_color;
    vec3 specular_color;
    uint specular_exponent;
    uint no_texture;
};

layout(set = 0, binding = 1) uniform uniforms {
    vec2 windowSize;
    vec3 viewPos;
    Style style;
};

void main() {
    //vec2 position = gl_FragCoord.xy / windowSize;
    //outColor = vec4(distance(position, vec2(0.5, 0)), distance(position, vec2(0, 1)), distance(position, vec2(1, 1)), 1.0);


    vec3 lightPos = vec3(1.2, 2.0, 0);
    vec3 lightColor = vec3(1.0, 1.0, 0.0);

    float ambientStrength = 0.02;
    vec3 ambient = ambientStrength * style.ambient_color;

    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(lightPos - fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * style.diffuse_color;

    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 10);
    vec3 specular = 0.5 * spec * style.specular_color;

    outColor = vec4(diffuse + specular + ambient, 1.0);
}
