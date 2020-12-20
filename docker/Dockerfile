FROM pytorch/pytorch:latest

RUN apt-get update -y
RUN apt-get install -y git cmake build-essential language-pack-ko
RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8
RUN export PATH="/usr/local/bin:$PATH"

RUN git clone https://github.com/kakao/khaiii.git
WORKDIR /workspace/khaiii

RUN mkdir build
WORKDIR /workspace/khaiii/build

RUN cmake ..
RUN make all
RUN make resource
