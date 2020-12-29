#!/bin/sh

COMMON_ARGS="--target-env vulkan1.1"

glslangValidator -V lighting.vert.glsl $COMMON_ARGS -o lighting.vert.spv || exit 1
glslangValidator -V lighting.frag.glsl $COMMON_ARGS -o lighting.frag.spv || exit 1

glslangValidator -V billboard.vert.glsl $COMMON_ARGS -o billboard.vert.spv || exit 1
glslangValidator -V billboard.frag.glsl $COMMON_ARGS -o billboard.frag.spv || exit 1
