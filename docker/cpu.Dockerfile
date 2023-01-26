FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive
ENV PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    g++ \
    make \
    cmake \
    wget \
    unzip \
    vim \
    git \
    libopencv-dev \
    libboost-all-dev \
    python3 \
    python3-pip \
    tmux

RUN git config --global advice.detachedHead false
RUN pip3 install --upgrade pip
RUN pip3 install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install pytest==7.2.0 onnx==1.13.0 flatbuffers==23.1.21 ranger-fm

COPY /lib /usr/local/lib
COPY /data /data
COPY /server /server
WORKDIR /server
RUN ./build.sh
CMD ./predict /data/mobile.onnx /data/labels.txt
