#!/bin/sh

COMMON_ARGS="--target-env vulkan1.1"

glslangValidator -V lighting.vert.glsl $COMMON_ARGS -o lighting.vert.spv || exit 1
glslangValidator -V lighting.frag.glsl $COMMON_ARGS -o lighting.frag.spv || exit 1
