# Stage - Build
FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime as builder

WORKDIR /khaiii

RUN apt-get update -y && \
    apt-get install -y language-pack-ko

RUN locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8

COPY requirements.txt .
RUN pip install -r requirements.txt

# Build khaiii
COPY README.md LICENSE CMakeLists.txt ./
COPY cmake/ ./cmake
COPY include/ ./include
COPY src/ ./src

WORKDIR /khaiii/build
RUN cmake /khaiii
RUN make all

# Build resources
COPY rsc/ /khaiii/rsc
RUN make large_resource && \
    mv ./share/khaiii ./share/large && \
    make resource


# Stage - Test
FROM builder AS test

CMD ["/khaiii/build/test/khaiii"]


# Stage - Runtime
FROM ubuntu:16.04 AS runtime

LABEL maintainer="krikit <krikit@naver.com>"

WORKDIR /khaiii

RUN apt-get update -y && \
    apt-get install -y language-pack-ko

RUN locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8

ENV PATH "$PATH:/khaiii/bin"

COPY --from=builder /khaiii/build/bin/ ./bin
COPY --from=builder /khaiii/build/lib/ ./lib
COPY --from=builder /khaiii/build/share/ ./share

ENTRYPOINT ["khaiii"]
CMD ["--rsc-dir=/khaiii/share/khaiii"]
