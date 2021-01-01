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
    bool use_parallax;
    bool use_ao;
} global_ubo;
layout (set = 1, binding = 1) uniform sampler2D textures_color[256];
layout (set = 1, binding = 2) uniform sampler2D textures_normal[256];
// x = displacement, y = roughness, z = metalness, w = ambient occlusion
layout (set = 1, binding = 3) uniform sampler2D textures_mat[256];
layout (set = 2, binding = 0, std140) uniform InstanceSpecificUBO {
  mat4 model;
} instance_ubo;

layout (location = 0) in VS_OUT {
  vec2 fragTexCoord;
  vec3 fragPos;
  vec3 tangent_view_pos;
  vec3 tangent_frag_pos;
  vec3 tangent_light_positions[4];
  flat uint texIdx;
} fs_in;

layout (location = 0) out vec4 outColor;

vec3 schlick_fresnel(in vec3 F0, in float cos_theta) {
  return F0 + (1.0 - F0) * pow(1.0 - cos_theta, 5.0);
}

float distribution_ggx(in float roughness, in float NdH) {
  float a = roughness * roughness;
  float a2 = a * a;
  float NdH2 = NdH * NdH;
  float num = a2;
  float denom = NdH2 * (a2 - 1.0) + 1.0;
  denom = 3.1415926 * denom * denom;
  return num / denom;
}

float schlick_ggx(in float roughness, in float NdV) {
  float r = (roughness + 1.0);
  float k = (r * r) / 8.0;

  float num = NdV;
  float denom = NdV * (1.0 - k) + k;
  return num / denom;
}

float smith_geometry(in float NdL, in float NdV, in float roughness) {
  float ggx1 = schlick_ggx(roughness, NdV);
  float ggx2 = schlick_ggx(roughness, NdL);
  return ggx1 * ggx2;
}

vec3 cooktorrance_specular(in float NdL, in float NdV, in float NDF, in float G, in vec3 F) {
  vec3 num = NDF * G * F;
  float denom = 4.0 * NdV * NdL;
  return num / max(denom, 0.001);
}

float get_height(in vec2 tex_coords) {
  return 1.0 - texture(textures_mat[nonuniformEXT(fs_in.texIdx)], tex_coords).r;
}

vec2 parallax_mapping(in vec3 view_dir) {
  vec2 tex_coords = fs_in.fragTexCoord;
  float displacement = get_height(tex_coords);
  float height_scale = 0.1;
  vec2 p = view_dir.xy / view_dir.z * (displacement * height_scale);
  return tex_coords - p;
}

void parallax_mapping_steep(in vec3 view_dir, out vec2 prev_tex_coords, out float prev_height_value, out vec2 current_tex_coords, out float current_height_value, out float layer_height, out float current_layer_height, out vec2 dtex) {
  const float height_scale = 0.01;
  const float min_layers = 8;
  const float max_layers = 32;
  float num_layers = mix(max_layers, min_layers, abs(dot(vec3(0.0, 0.0, 1.0), view_dir)));
  layer_height = 1.0 / num_layers;
  current_layer_height = 0.0;
  dtex = height_scale * view_dir.xy / view_dir.z / num_layers;

  current_tex_coords = fs_in.fragTexCoord;
  prev_tex_coords = fs_in.fragTexCoord;
  current_height_value = get_height(current_tex_coords);
  prev_height_value = current_height_value;

  while(current_height_value > current_layer_height) {
    prev_tex_coords = current_tex_coords;
    prev_height_value = current_height_value;

    current_layer_height += layer_height;
    current_tex_coords -= dtex;
    current_height_value = get_height(current_tex_coords);
  }
}

vec2 parallax_mapping_occlusion(in vec3 view_dir) {
  vec2 prev_tex_coords = vec2(0.0);
  float prev_height_value = 0.0;
  vec2 current_tex_coords = vec2(0.0);
  float current_height_value = 0.0;
  float layer_height = 0.0;
  float current_layer_height = 0.0;
  vec2 dtex = vec2(0.0);
  parallax_mapping_steep(view_dir, prev_tex_coords, prev_height_value, current_tex_coords, current_height_value, layer_height, current_layer_height, dtex);

  float next_height = current_height_value - current_layer_height;
  float prev_height = prev_height_value - current_layer_height + layer_height;

  float weight = next_height / (next_height - prev_height);
  vec2 final_tex_coords = prev_tex_coords * weight + current_tex_coords * (1.0 - weight);
  return final_tex_coords;
}

vec2 parallax_mapping_relief(in vec3 view_dir) {
  vec2 prev_tex_coords = vec2(0.0);
  float prev_height_value = 0.0;
  vec2 current_tex_coords = vec2(0.0);
  float current_height_value = 0.0;
  float layer_height = 0.0;
  float current_layer_height = 0.0;
  vec2 dtex = vec2(0.0);
  parallax_mapping_steep(view_dir, prev_tex_coords, prev_height_value, current_tex_coords, current_height_value, layer_height, current_layer_height, dtex);

  vec2 delta_tex_coords = dtex / 2;
  float delta_height = layer_height / 2;

  current_tex_coords += delta_tex_coords;
  current_layer_height -= delta_height;

  const int num_search_steps = 5;
  for(int i = 0; i < num_search_steps; i++) {
    delta_tex_coords /= 2;
    delta_height /= 2;

    current_height_value = get_height(current_tex_coords);

    if(current_height_value > current_layer_height) {
      current_tex_coords -= delta_tex_coords;
      current_layer_height += delta_height;
    } else {
      current_tex_coords += delta_tex_coords;
      current_layer_height -= delta_height;
    }
  }

  return current_tex_coords;
}

void main() {
  vec3 frag_pos = fs_in.tangent_frag_pos;
  vec3 view_pos = fs_in.tangent_view_pos;
  vec3 V = normalize(view_pos - frag_pos);
  vec2 tex_coords = fs_in.fragTexCoord;
  if (global_ubo.use_parallax) {
    tex_coords = parallax_mapping_relief(V);
    if(tex_coords.x > 1.0 || tex_coords.y > 1.0 || tex_coords.x < 0.0 || tex_coords.y < 0.0) {
      discard;
    }
  }

  vec3 albedo = texture(textures_color[nonuniformEXT(fs_in.texIdx)], tex_coords).xyz;
  vec3 N = texture(textures_normal[nonuniformEXT(fs_in.texIdx)], tex_coords).xyz;
  N = normalize(N * 2.0 - 1.0);
  vec4 mat_props = texture(textures_mat[nonuniformEXT(fs_in.texIdx)], tex_coords);
  float roughness = mat_props.g;
  float metallic = mat_props.b;
  float ao = 1.0;
  if (global_ubo.use_ao) {
    ao = mat_props.a;
  }

  vec3 Lo = vec3(0.0);
  for(int i  = 0; i < 4; i++) {
    vec3 light_pos = fs_in.tangent_light_positions[i];
    vec3 L = normalize(light_pos - frag_pos);
    vec3 H = normalize(V + L);
    float NdV = max(dot(N, V), 0.0);
    float NdL = max(dot(N, L), 0.0);
    float NdH = max(dot(N, H), 0.0);

    float distance = length(light_pos - frag_pos);
    float attenuation = 1.0 / (distance * distance);
    vec3 radiance = global_ubo.light_colors[i].xyz * attenuation;

    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);
    vec3 F = schlick_fresnel(F0, max(dot(H, V), 0.0));
    float NDF = distribution_ggx(roughness, NdH);
    float G = smith_geometry(NdL, NdV, roughness);
    vec3 specular = cooktorrance_specular(NdL, NdV, NDF, G, F);

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;
    Lo += (kD * albedo / 3.1415926 + specular) * radiance * NdL;
  }

  vec3 ambient = vec3(0.03) * albedo * ao;
  vec3 color = clamp(ambient + Lo, vec3(0.0), vec3(1.0));
  outColor = vec4(color, 1.0);
}
