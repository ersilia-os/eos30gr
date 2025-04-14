FROM bentoml/model-server:0.11.0-py311
MAINTAINER ersilia


RUN pip install deepchem==2.8.1.dev20240705204934
RUN pip install tqdm==4.66.4
RUN pip install torch==2.5.1


WORKDIR /repo
COPY . /repo
