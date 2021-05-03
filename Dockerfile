FROM tensorflow/tensorflow:2.4.1

ENV SERVICE_NAME=style-transfer
ENV PROJECT_DIRECTORY=/opt/${SERVICE_NAME}

RUN mkdir -p ${PROJECT_DIRECTORY}
WORKDIR ${PROJECT_DIRECTORY}

RUN apt-get update && \
    apt-get install -y \
    git \
    wget \
    nano \
    curl

ADD . .

RUN python -m pip install -U pip && \
    python -m pip install -r requirements.txt

CMD python service/run_server.py
