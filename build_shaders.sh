#!/bin/sh

glslangValidator -V lighting.vert.glsl -o lighting.vert.spv || exit 1
glslangValidator -V lighting.frag.glsl -o lighting.frag.spv || exit 1
