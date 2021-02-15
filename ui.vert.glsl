
#version 450

#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in vec2 pos;
layout (location = 1) in vec2 uv;
layout (location = 2) in vec4 color;

layout (location = 0) out VS_OUT {
  vec2 pos;
  vec2 uv;
  vec4 color;
} vs_out;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = vec4(pos, 0.0, 0.0);
    vs_out.pos = pos;
    vs_out.uv = uv;
    vs_out.color = color;
}
