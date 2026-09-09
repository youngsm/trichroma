#!/bin/bash

set -e

docker_dirs="chroma3.deps chroma3.435 chroma3.440 chroma3.latest chroma3.nvidia.base chroma3.nvidia"
#docker_dirs="chroma3.nvidia.base chroma3.nvidia"

for dir in $docker_dirs; do
    image=$(echo $dir | sed 's/chroma3./benland100\/chroma3:/')
    echo $image
    docker build -t $image $dir
    docker push $image
done

for sing in chroma3.*/Singularity; do
    dir=$(dirname $sing)
    cd $dir
    image="../$dir.simg"
    echo $image
    singularity build $image Singularity
    cd ..
done
