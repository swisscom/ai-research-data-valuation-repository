FROM pytorch/pytorch:latest
RUN apt-get update && apt-get install python3-pip -y
COPY . /opt/app
WORKDIR /opt/app
RUN python3 -m pip install -r requirements.txt
RUN python3 -m space download en
