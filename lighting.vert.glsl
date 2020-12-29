
#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

layout (set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec4 view_pos;
    vec4 view_dir;
    bool use_diffuse;
    bool use_specular;
} camera;

layout (set = 2, binding = 0, std140) uniform InstanceSpecificUBO {
  mat4 model;
} instance_ubo;

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec3 inTangent;
layout (location = 3) in vec2 inTexCoord;
layout (location = 4) in uint inTexIdx;

layout (location = 0) out vec2 fragTexCoord;
layout (location = 1) out vec3 normal;
layout (location = 2) out vec3 tangent;
layout (location = 3) out vec3 fragPos;
layout (location = 4) out flat uint texIdx;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = camera.proj * camera.view * instance_ubo.model * vec4(inPosition, 1.0);
    fragTexCoord = inTexCoord;
    normal = inNormal;
    tangent = inTangent;
    fragPos = inPosition;
    texIdx = nonuniformEXT(inTexIdx);
}
