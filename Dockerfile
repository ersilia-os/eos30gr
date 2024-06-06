FROM bentoml/model-server:0.11.0-py311
MAINTAINER ersilia

RUN pip install rdkit==2022.9.6
RUN pip install lazyqsar==0.4



WORKDIR /repo
COPY . /repo
