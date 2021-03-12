#version 450

layout (input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput src_frame;

layout (location = 0) in vec2 tex_coords;

layout (location = 0) out vec4 frag_color;

layout (set = 1, binding = 0, std140) uniform UniformBufferObject {
    float exposure;
    float white_luminance;
    uint algo;
} control;

const uint ALGO_NOOP = 0;
const uint ALGO_LINEAR = 1;
const uint ALGO_REINHARD_SIMPLE = 2;
const uint ALGO_REINHARD_ENHANCED = 3;
const uint ALGO_UNCHARTED = 4;
const uint ALGO_ACES = 5;

vec3 exposure(vec3 in_color) {
  if(control.algo == ALGO_NOOP) {
    return in_color;
  } else if(control.algo == ALGO_LINEAR) {
    return in_color * exp2(control.exposure);
  } else if(control.algo == ALGO_REINHARD_SIMPLE) {
    return in_color * 2.0 * exp2(control.exposure);
  } else if(control.algo == ALGO_REINHARD_ENHANCED) {
    return in_color * 2.0 * exp2(control.exposure);
  } else if(control.algo == ALGO_UNCHARTED) {
    return in_color * 4.0 * exp2(control.exposure);
  } else if(control.algo == ALGO_ACES) {
    return in_color * exp2(control.exposure);
  } else {
    return in_color;
  }
}

vec3 algo_linear(vec3 in_color) {
  vec3 out_color = clamp(in_color, 0.0, 1.0);
  return out_color;
}

vec3 algo_reinhard_simple(vec3 in_color) {
  vec3 out_color = in_color / (vec3(1.0) + in_color);
  return out_color;
}

vec3 algo_reinhard_enhanced(vec3 in_color) {
  vec3 white_squared = vec3(control.white_luminance * control.white_luminance);
  vec3 out_color = (in_color * (1.0 + in_color / white_squared)) / (1.0 + in_color);
  return out_color;
}

// http://filmicworlds.com/blog/filmic-tonemapping-operators/
vec3 uncharted_tonemap(vec3 color) {
  const float A = 0.15;
  const float B = 0.50;
  const float C = 0.10;
  const float D = 0.20;
  const float E = 0.02;
  const float F = 0.30;

  return (
	  (color * (A * color + C * B) + D * E) /
	  (color * (A * color + B) + D * F)
	  ) - E / F;
}

vec3 algo_uncharted(vec3 in_color) {
  float W = 11.2;

  vec3 tonemapped = uncharted_tonemap(in_color);
  vec3 white_scale = 1.0 / uncharted_tonemap(vec3(W));
  vec3 out_color = tonemapped * white_scale;
  return out_color;
}

vec3 algo_aces(vec3 in_color) {
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;

  vec3 numerator = in_color * (a * in_color + b);
  vec3 denominator = in_color * (c * in_color + d) + e;

  return numerator / denominator;
}

void main() {
  vec3 src_color = exposure(subpassLoad(src_frame).rgb);
  vec3 out_color = src_color;
  if(control.algo == ALGO_NOOP) {
    out_color = src_color;
  } else if(control.algo == ALGO_LINEAR) {
    out_color = algo_linear(src_color);
  } else if(control.algo == ALGO_REINHARD_SIMPLE) {
    out_color = algo_reinhard_simple(src_color);
  } else if(control.algo == ALGO_REINHARD_ENHANCED) {
    out_color = algo_reinhard_enhanced(src_color);
  } else if(control.algo == ALGO_UNCHARTED) {
    out_color = algo_uncharted(src_color);
  } else if(control.algo == ALGO_ACES) {
    out_color = algo_aces(src_color);
  } else {
    out_color = vec3(1.0, 0.411, 0.706);
  }
  frag_color = vec4(out_color, 1.0);
}
