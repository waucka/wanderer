
#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

layout (set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 view_pos;
    bool use_diffuse;
    bool use_specular;
} camera;

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inTexCoord;
layout (location = 3) in uint inTexIdx;

layout (location = 0) out vec2 fragTexCoord;
layout (location = 1) out vec3 normal;
layout (location = 2) out vec3 fragPos;
layout (location = 3) out flat uint texIdx;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = camera.proj * camera.view * camera.model * vec4(inPosition, 1.0);
    fragTexCoord = inTexCoord;
    normal = inNormal;
    fragPos = inPosition;
    texIdx = nonuniformEXT(inTexIdx);
}
