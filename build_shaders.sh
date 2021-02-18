#!/bin/sh

COMMON_ARGS="--target-env vulkan1.2"

glslangValidator -V lighting.vert.glsl $COMMON_ARGS -o lighting.vert.spv || exit 1
glslangValidator -V lighting.frag.glsl $COMMON_ARGS -o lighting.frag.spv || exit 1

glslangValidator -V hdr.vert.glsl $COMMON_ARGS -o hdr.vert.spv || exit 1
glslangValidator -V hdr.frag.glsl $COMMON_ARGS -o hdr.frag.spv || exit 1

glslangValidator -V ui.vert.glsl $COMMON_ARGS -o ui.vert.spv || exit 1
glslangValidator -V ui.frag.glsl $COMMON_ARGS -o ui.frag.spv || exit 1
