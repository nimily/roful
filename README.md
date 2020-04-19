# Introduction
Implementation of various instances of ROFUL.


# Docker
Build `roful` image using the following command:
```shell script
docker build -t roful .
```
Then, run the image through:
```shell script
PLOT_SRC="$(pwd)/plots"
PLOT_TAR="/root/roful/plots"
docker run -t --rm --mount type=bind,source="$PLOT_SRC",target="$PLOT_TAR" roful
```
