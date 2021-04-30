#!/bin/sh

COMMON_ARGS="--target-env vulkan1.2"

mkdir -p ./assets/shaders

glslangValidator -V lighting.vert.glsl $COMMON_ARGS -o ./assets/shaders/lighting.vert.spv || exit 1
glslangValidator -V lighting.frag.glsl $COMMON_ARGS -o ./assets/shaders/lighting.frag.spv || exit 1

glslangValidator -V hdr.vert.glsl $COMMON_ARGS -o ./assets/shaders/hdr.vert.spv || exit 1
glslangValidator -V hdr.frag.glsl $COMMON_ARGS -o ./assets/shaders/hdr.frag.spv || exit 1

glslangValidator -V ui.vert.glsl $COMMON_ARGS -o ./assets/shaders/ui.vert.spv || exit 1
glslangValidator -V ui.frag.glsl $COMMON_ARGS -o ./assets/shaders/ui.frag.spv || exit 1

glslangValidator -V star.vert.glsl $COMMON_ARGS -o ./assets/shaders/star.vert.spv || exit 1
glslangValidator -V star.frag.glsl $COMMON_ARGS -o ./assets/shaders/star.frag.spv || exit 1
