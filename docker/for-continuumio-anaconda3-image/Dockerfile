FROM continuumio/anaconda3
MAINTAINER byeongho.kang@yahoo.com

ENV PATH="/opt/conda/bin:${PATH}"

RUN apt -y update
RUN apt -y install build-essential locales locales-all

WORKDIR /root
RUN git clone https://github.com/kakao/khaiii.git

WORKDIR /root/khaiii
RUN pip install -r requirements.txt

WORKDIR /root/khaiii/build
RUN cmake ..
RUN make all
RUN make resource
RUN make install

RUN make package_python
WORKDIR /root/khaiii/build/package_python
RUN pip install .

RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8

WORKDIR /root/khaiii/build
