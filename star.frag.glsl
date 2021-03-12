#version 450

#extension GL_ARB_gpu_shader_int64 : enable

layout (location = 0) out vec4 frag_color;
//layout (depth_any) out float gl_FragDepth;
in vec4 gl_FragCoord;

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

layout (set = 1, binding = 0, std140) uniform StarInfo {
  mat4 transform;
  // Should this be calculated via transform * vec4(0.0)?  Would having a single
  // point of truth be worth the performance hit?
  vec4 center;
  float radius;
  float luminosity;
  uint temperature;
  uint heat_color_lower_bound;
  uint heat_color_upper_bound;
} star_info;

layout (set = 1, binding = 1) uniform sampler1D heat_color;

layout (location = 0) in FS_IN {
  vec3 fragPos;
} fs_in;

// By Morgan McGuire @morgan3d, http://graphicscodex.com
// Reuse permitted under the BSD license.

// All noise functions are designed for values on integer scale.
// They are tuned to avoid visible periodicity for both positive and
// negative coordinates within a few orders of magnitude.

// For a single octave
//#define NOISE noise

// For multiple octaves
#define NOISE fbm
#define NUM_NOISE_OCTAVES 5

// Precision-adjusted variations of https://www.shadertoy.com/view/4djSRW
float hash(float p) { p = fract(p * 0.011); p *= p + 7.5; p *= p + p; return fract(p); }
float hash(vec2 p) {vec3 p3 = fract(vec3(p.xyx) * 0.13); p3 += dot(p3, p3.yzx + 3.333); return fract((p3.x + p3.y) * p3.z); }

float noise(float x) {
  float i = floor(x);
  float f = fract(x);
  float u = f * f * (3.0 - 2.0 * f);
  return mix(hash(i), hash(i + 1.0), u);
}


float noise(vec2 x) {
  vec2 i = floor(x);
  vec2 f = fract(x);

  // Four corners in 2D of a tile
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));

  // Simple 2D lerp using smoothstep envelope between the values.
  // return vec3(mix(mix(a, b, smoothstep(0.0, 1.0, f.x)),
  //			mix(c, d, smoothstep(0.0, 1.0, f.x)),
  //			smoothstep(0.0, 1.0, f.y)));

  // Same code, with the clamps in smoothstep and common subexpressions
  // optimized away.
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}


float noise(vec3 x) {
  const vec3 step = vec3(110, 241, 171);

  vec3 i = floor(x);
  vec3 f = fract(x);
 
  // For performance, compute the base input to a 1D hash from the integer part of the argument and the 
  // incremental change to the 1D based on the 3D -> 1D wrapping
  float n = dot(i, step);

  vec3 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix( hash(n + dot(step, vec3(0, 0, 0))), hash(n + dot(step, vec3(1, 0, 0))), u.x),
                 mix( hash(n + dot(step, vec3(0, 1, 0))), hash(n + dot(step, vec3(1, 1, 0))), u.x), u.y),
             mix(mix( hash(n + dot(step, vec3(0, 0, 1))), hash(n + dot(step, vec3(1, 0, 1))), u.x),
                 mix( hash(n + dot(step, vec3(0, 1, 1))), hash(n + dot(step, vec3(1, 1, 1))), u.x), u.y), u.z);
}

float fbm(float x) {
  float v = 0.0;
  float a = 0.5;
  float shift = float(100);
  for (int i = 0; i < NUM_NOISE_OCTAVES; ++i) {
    v += a * noise(x);
    x = x * 2.0 + shift;
    a *= 0.5;
  }
  return v;
}

float fbm(vec2 x) {
  float v = 0.0;
  float a = 0.5;
  vec2 shift = vec2(100);
  // Rotate to reduce axial bias
  mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.50));
  for (int i = 0; i < NUM_NOISE_OCTAVES; ++i) {
    v += a * noise(x);
    x = rot * x * 2.0 + shift;
    a *= 0.5;
  }
  return v;
}

float fbm(vec3 x) {
  float v = 0.0;
  float a = 0.5;
  vec3 shift = vec3(100);
  for (int i = 0; i < NUM_NOISE_OCTAVES; ++i) {
    v += a * noise(x);
    x = x * 2.0 + shift;
    a *= 0.5;
  }
  return v;
}

const float pi          = 3.1415926535;
const float inf         = 1.0 / 0.0;
float square(float x) { return x * x; }
float infIfNegative(float x) { return (x >= 0.0) ? x : inf; }

// C = sphere center, r = sphere radius, P = ray origin, w = ray direction
float  intersectSphere(vec3 C, float r, vec3 P, vec3 w) {	
  vec3 v = P - C;
  float b = -dot(w, v);
  float c = dot(v, v) - square(r);
  float d = (square(b) - c);
  if (d < 0.0) { return inf; }	
  float dsqrt = sqrt(d);
	
  // Choose the first positive intersection
  return min(infIfNegative((b - dsqrt)), infIfNegative((b + dsqrt)));
}

// End BSD-licensed noise implementation

void main() {
  float current_time = camera.current_time;
  vec3 direction = normalize(fs_in.fragPos - camera.view_pos.xyz);
  // TODO: add some noise to the radius
  float intersection_distance = intersectSphere(star_info.center.xyz,
                                                star_info.radius,
                                                camera.view_pos.xyz,
                                                direction);
  if (!(intersection_distance < inf)) {
    gl_FragDepth = gl_FragCoord.z;
    discard;
  }
  vec3 hit_pos = camera.view_pos.xyz + direction * intersection_distance;
  gl_FragDepth = gl_FragCoord.z;
  // TODO: figure out why this doesn't work
  //gl_FragDepth = (camera.proj * camera.view * vec4(hit_pos, 1.0)).z;
  uint temperature_range = star_info.heat_color_upper_bound - star_info.heat_color_lower_bound;
  uint temperature_index_int = star_info.temperature - star_info.heat_color_lower_bound;
  float temperature_index = float(temperature_index_int) / float(temperature_range);
  temperature_index = clamp(temperature_index,
                            0.0,
                            1.0);
  vec4 color = texture(heat_color, temperature_index);
  color.r *= star_info.luminosity;
  color.g *= star_info.luminosity;
  color.b *= star_info.luminosity;

  vec3 sphere_pos = hit_pos - star_info.center.xyz;
  float theta = atan(sphere_pos.y, sphere_pos.x);
  theta += ((camera.current_time / 100.0) * pi) / star_info.radius;
  float phi = acos(sphere_pos.z / sqrt(square(sphere_pos.x) + square(sphere_pos.y) + square(sphere_pos.z)));
  float noise_scale = 30.0 * log(star_info.radius + 1.0);
  float adjustment = NOISE(vec2(theta * noise_scale, phi * noise_scale));

  frag_color = color * adjustment;
  frag_color.a = 1.0;
  // DEBUG
  //frag_color = vec4(vec3(100.0), 1.0);
}
