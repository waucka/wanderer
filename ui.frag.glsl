#version 450

#extension GL_ARB_separate_shader_objects : enable

layout (set = 0, binding = 0) uniform sampler2D ui_texture;

layout (location = 0) in VS_OUT {
  vec2 pos;
  vec2 uv;
  vec4 color;
} fs_in;

layout (location = 0) out vec4 out_color;

void main() {
  vec4 tex_color = texture(ui_texture, fs_in.uv);
  out_color = tex_color * fs_in.color;
}
