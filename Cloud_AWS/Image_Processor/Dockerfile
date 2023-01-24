FROM ubuntu:latest

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update
RUN apt-get install -y python3 python3-pip mosquitto-clients
RUN pip3 install Cython
RUN pip3 install paho-mqtt
RUN pip3 install numpy

RUN pip3 install boto3
COPY imageprocessor.py imageprocessor.py

CMD ["python3","imageprocessor.py"]
