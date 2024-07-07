FROM bentoml/model-server:0.11.0-py311
MAINTAINER ersilia


RUN pip install deepchem==2.8.1.dev20240705204934
RUN pip install tqdm==4.66.4
RUN conda install -c conda-forge
RUN pip install torch


WORKDIR /repo
COPY . /repo
