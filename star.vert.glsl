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
    uvec4 current_time;
    bool use_parallax;
    bool use_ao;
} camera;

layout (set = 1, binding = 0, std140) uniform StarInfo {
  mat4 transform;
  // Should this be calculated via transform * vec4(0.0)?  Would having a single
  // point of truth be worth the performance hit?
  vec4 center;
  float radius;
  uint temperature;
  uint heat_color_lower_bound;
  uint heat_color_upper_bound;
} star_info;

layout (location = 0) in vec3 inPosition;

layout (location = 0) out VS_OUT {
  vec3 fragPos;
} vs_out;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = camera.proj * camera.view * star_info.transform * vec4(inPosition, 1.0);
    vec3 fragPos = (star_info.transform * vec4(inPosition, 1.0)).xyz;
    vs_out.fragPos = fragPos;
}
