
#version 450

#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in vec2 pos;
layout (location = 1) in vec2 uv;
layout (location = 2) in uvec4 color;


layout (set = 0, binding = 0, std140) uniform UIUBO {
  vec2 screen_size;
} ui_ubo;

layout (location = 0) out VS_OUT {
  vec2 pos;
  vec2 uv;
  vec4 color;
} vs_out;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
  float scaled_x = ((pos.x / ui_ubo.screen_size.x) - 0.5) * 2.0;
  float scaled_y = ((pos.y / ui_ubo.screen_size.y) - 0.5) * 2.0;
  vec2 fixed_pos = vec2(scaled_x, scaled_y);
  gl_Position = vec4(fixed_pos, 0.5, 1.0);
  vs_out.pos = fixed_pos;
  vs_out.uv = uv;
  vs_out.color = vec4(color.r / 255.0, color.g / 255.0, color.b / 255.0, color.a / 255.0);
}
