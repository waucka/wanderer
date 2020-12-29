#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

layout (set = 0, binding = 0, std140) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec4 view_pos;
    vec4 view_dir;
    bool use_diffuse;
    bool use_specular;
} global_ubo;
layout (set = 1, binding = 1) uniform sampler2D textures_color[256];
layout (set = 1, binding = 2) uniform sampler2D textures_normal[256];
// x = displacement, y = roughness, z = metalness, w = ambient occlusion?
layout (set = 1, binding = 3) uniform sampler2D textures_mat[256];
layout (set = 2, binding = 0, std140) uniform InstanceSpecificUBO {
  mat4 model;
} instance_ubo;

layout (location = 0) in vec2 fragTexCoord;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 tangent;
layout (location = 3) in vec3 fragPos;
layout (location = 4) in flat uint texIdx;

layout (location = 0) out vec4 outColor;

vec3 fresnel_factor(in vec3 f0, in float product) {
  return mix(f0, vec3(1.0), pow(1.01 - product, 5.0));
}

float D_GGX(in float roughness, in float NdH) {
  float m = roughness * roughness;
  float m2 = m * m;
  float d = (NdH * m2 - NdH) * NdH + 1.0;
  return m2 / (3.1415926 * d * d);
}

float G_schlick(in float roughness, in float NdV, in float NdL) {
  float k = roughness * roughness * 0.5;
  float V = NdV * (1.0 - k) + k;
  float L = NdL * (1.0 - k) + k;
  return 0.25 / (V * L);
}

vec3 cooktorrance_specular(in float NdL, in float NdV, in float NdH, in vec3 specular, in float roughness) {
  float D = D_GGX(roughness, NdH);
  float G = G_schlick(roughness, NdV, NdL);
  //float rim = mix(1.0 - roughness * rim_factor * 0.9, 1.0, NdV);
  return specular * G * D;
}

void main() {
  vec3 base = texture(textures_color[nonuniformEXT(texIdx)], fragTexCoord).xyz;
  vec3 norm = texture(textures_normal[nonuniformEXT(texIdx)], fragTexCoord).xyz;
  vec4 mat_props = texture(textures_mat[nonuniformEXT(texIdx)], fragTexCoord);
  //vec3 base = texture(textures_color[0], fragTexCoord).xyz;
  //vec3 norm = texture(textures_normal[0], fragTexCoord).xyz;
  //vec4 mat_props = texture(textures_mat[0], fragTexCoord);
  //vec3 base = vec3(1.0);
  //vec3 norm = vec3(0.0, 0.0, 1.0);
  //vec4 mat_props = vec4(1.0);

  vec3 v_pos = (global_ubo.view * instance_ubo.model * vec4(fragPos, 1.0)).xyz;
  vec3 lightPos = vec3(0.0, 0.0, 5.0);
  vec3 lightColorBase = vec3(1.0, 1.0, 1.0);
  float ambientStrength = 0.01;
  vec3 view_light_pos = (global_ubo.view * vec4(lightPos, 1.0)).xyz;
  float phong_diffuse = 0.318309;

  float A = 20.0 / dot(view_light_pos - v_pos, view_light_pos - v_pos);
  vec3 L = normalize(view_light_pos - v_pos);
  vec3 V = normalize(-v_pos);
  vec3 H = normalize(L + V);
  vec3 nn = normalize(normal);
  vec3 nt = normalize(tangent);
  mat3x3 tbn = mat3x3(nt, cross(nn, nt), nn);
  vec3 N = tbn * (norm * 2.0 - 1.0);

  float roughness = mat_props.y;
  float metallic = mat_props.z;

  // 0.04 was found online; is that really a good specular value for
  // a non-metallic object?
  vec3 specular = mix(vec3(0.04), base, metallic);

  float NdL = max(0.0, dot(N, L));
  float NdV = max(0.001, dot(N, V));
  float NdH = max(0.001, dot(N, H));
  float HdV = max(0.001, dot(H, V));
  float LdV = max(0.001, dot(L, V));

  vec3 specfresnel = fresnel_factor(specular, HdV);
  vec3 specref = cooktorrance_specular(NdL, NdV, NdH, specfresnel, roughness);

  specref *= vec3(NdL);
  vec3 diffref = (vec3(1.0) - specfresnel) * phong_diffuse * NdL;

  vec3 reflected_light = vec3(0);
  vec3 diffuse_light = vec3(0);
  vec3 light_color = lightColorBase * A;
  reflected_light += specref * light_color;
  diffuse_light += diffref * light_color;
  vec3 lighting = vec3(ambientStrength);
  if (global_ubo.use_specular) {
    lighting += reflected_light;
  }
  if (global_ubo.use_diffuse) {
    lighting += diffuse_light;
  }
  outColor = vec4(lighting, 1.0);
}
