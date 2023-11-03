# FROM ubuntu:23.10
FROM python:3.11.4
COPY . /digits/
# RUN apt-get update
# RUN apt-get install -y python3 python3-pip
RUN pip3 install -r /digits/requirements.txt
WORKDIR /digits
CMD ["python","digits.py"]