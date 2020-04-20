FROM continuumio/miniconda3

WORKDIR /root/roful

COPY . ./

RUN conda install numpy=1.18.1 matplotlib=3.1.3

ENV n=25 k=100 d=50 t=1000 s=1 g=0

CMD PYTHONPATH=src python -m experiments -n $n -k $k -d $d -t $t -s $s -g $g