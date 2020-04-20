FROM continuumio/miniconda3

WORKDIR /root/roful

COPY . ./

RUN conda install numpy=1.18.1 matplotlib=3.1.3

ARG n=25
ARG k=100
ARG d=50
ARG t=1000
ARG s=1
ARG g=0

CMD PYTHONPATH=src python -m experiments -n $n -k $k -d $d -t $t -s $s -g $g