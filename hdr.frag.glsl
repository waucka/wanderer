#version 450

layout (input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput src_frame;

layout (location = 0) in vec2 tex_coords;

layout (location = 0) out vec4 frag_color;

layout (set = 0, binding = 1, std140) uniform UniformBufferObject {
    float exposure;
    float gamma;
    uint algo;
} control;

const uint ALGO_NOOP = 0;
const uint ALGO_LINEAR = 1;
const uint ALGO_REINHARD_SIMPLE = 2;

vec3 algo_linear(vec3 in_color) {
  vec3 out_color = clamp(control.exposure * in_color, 0.0, 1.0);
  out_color = pow(out_color, vec3(1.0 / control.gamma));
  return out_color;
}

vec3 algo_reinhard_simple(vec3 in_color) {
  vec3 out_color = in_color * control.exposure / (1.0 + in_color / control.exposure);
  out_color = pow(out_color, vec3(1.0 / control.gamma));
  return out_color;
}

void main() {
  vec3 src_color = subpassLoad(src_frame).rgb;
  vec3 out_color = src_color;
  if(control.algo == ALGO_NOOP) {
    out_color = src_color;
  } else if(control.algo == ALGO_LINEAR) {
    out_color = algo_linear(src_color);
  } else if(control.algo == ALGO_REINHARD_SIMPLE) {
    out_color = algo_reinhard_simple(src_color);
  } else {
    out_color = vec3(1.0, 0.411, 0.706);
  }
  frag_color = vec4(out_color, 1.0);
}
