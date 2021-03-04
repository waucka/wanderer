#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

layout (set = 0, binding = 0, std140) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec4 view_pos;
    vec4 view_dir;
    vec4 light_positions[4];
    vec4 light_colors[4];
    float current_time;
    bool use_parallax;
    bool use_ao;
} camera;

layout (set = 2, binding = 0, std140) uniform InstanceSpecificUBO {
  mat4 model;
} instance_ubo;

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec3 inTangent;
layout (location = 3) in vec2 inTexCoord;
layout (location = 4) in uint inTexIdx;

layout (location = 0) out VS_OUT {
  vec2 fragTexCoord;
  vec3 fragPos;
  vec3 tangent_view_pos;
  vec3 tangent_frag_pos;
  vec3 tangent_light_positions[4];
  flat uint texIdx;
} vs_out;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = camera.proj * camera.view * instance_ubo.model * vec4(inPosition, 1.0);
    vs_out.fragTexCoord = inTexCoord;
    vec3 fragPos = (instance_ubo.model * vec4(inPosition, 1.0)).xyz;
    vs_out.fragPos = fragPos;

    vec3 T = normalize(vec3(instance_ubo.model * vec4(inTangent, 0.0)));
    vec3 N = normalize(vec3(instance_ubo.model * vec4(inNormal, 0.0)));
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);
    //vec3 B = normalize(vec3(instance_ubo.model * vec4(inBitangent, 0.0)));

    mat3 tbn = transpose(mat3(T, B, N));
    vs_out.tangent_view_pos = tbn * camera.view_pos.xyz;
    vs_out.tangent_frag_pos = tbn * fragPos;
    for (int i = 0; i < 4; i++) {
      vs_out.tangent_light_positions[i] = tbn * camera.light_positions[i].xyz;
    }
    vs_out.texIdx = nonuniformEXT(inTexIdx);
}
