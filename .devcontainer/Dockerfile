FROM ubuntu:20.10
RUN apt-get -y update
RUN apt-get install -y libgl1-mesa-dev && \
rm -rf /var/lib/apt/lists/*
RUN apt-get -y update
RUN apt-get install -y python3-pip
RUN apt-get install -y poppler-utils
RUN apt-get install -y libglib2.0-0
RUN pip install mediapipe