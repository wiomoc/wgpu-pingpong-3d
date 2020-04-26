#version 450

layout(location = 0) in vec3 vertexPos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texturePos;
layout(location = 3) in uint style;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 normalOut;
layout(location = 2) out vec2 texturePosOut;
layout(location = 3) flat out uint styleOut;

out gl_PerVertex {
    vec4 gl_Position;
};
layout(set = 0, binding = 1) uniform uniforms {
    mat4 viewProjectionMat;
    mat4 modelMat;
    mat4 normalMat;
};

void main() {
    fragPos =  vec3(modelMat * vec4(vertexPos, 1.0));
    normalOut = vec3(normalMat * vec4(normal, 1.0));
    texturePosOut = texturePos;
    styleOut = style;
    gl_Position = viewProjectionMat * modelMat * vec4(vertexPos, 1.0);
}
