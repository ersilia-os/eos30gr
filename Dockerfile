FROM bentoml/model-server:0.11.0-py37
MAINTAINER ersilia

RUN conda install -c conda-forge rdkit=2021.03.4
RUN pip install joblib==1.1.0
RUN conda install -c conda-forge keras=2.8.0
RUN conda install -c conda-forge tensorflow=2.8.1
RUN pip install gensim==3.8.3
RUN pip install git+https://github.com/samoturk/mol2vec

WORKDIR /repo
COPY . /repo
