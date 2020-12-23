#version 450

#extension GL_ARB_separate_shader_objects : enable

layout (set = 0, binding = 0, std140) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 view_pos;
    bool use_diffuse;
    bool use_specular;
} global_ubo;
layout (set = 1, binding = 1) uniform sampler2D textures[];
layout (set = 2, binding = 0, std140) uniform InstanceSpecificUBO {
  vec4 tint;
} instance_ubo;

layout (location = 0) in vec2 fragTexCoord;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 fragPos;

layout (location = 0) out vec4 outColor;

void main() {
  vec3 lightColor = vec3(1.0, 1.0, 1.0);
  float ambientStrength = 0.01;

  vec3 lightPos = vec3(0.0, 0.0, 2.0);
  vec3 norm = normalize(normal);
  vec3 lightDir = normalize(lightPos - fragPos);
  float diff = max(dot(norm, lightDir), 0.0);
  vec3 diffuse = diff * lightColor;

  float specularStrength = 0.5;
  vec3 viewDir = normalize(global_ubo.view_pos.xyz - fragPos);
  vec3 reflectDir = reflect(-lightDir, norm);
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
  vec3 specular = specularStrength * spec * lightColor;

  vec3 ambient = ambientStrength * lightColor;
  vec3 lighting = ambient;
  if (global_ubo.use_diffuse) {
    lighting = lighting + diffuse;
  }
  if (global_ubo.use_specular) {
    lighting = lighting + specular;
  }
  outColor = vec4(lighting, 1.0) * texture(textures[0], fragTexCoord);
}
