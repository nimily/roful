FROM continuumio/miniconda3

WORKDIR /root/roful

COPY . ./

RUN conda install numpy=1.18.1 matplotlib=3.1.3

CMD PYTHONPATH=src python -m experiments