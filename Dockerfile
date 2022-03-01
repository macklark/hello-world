FROM python:3

COPY ./requirments.txt .

RUN pip install -r requirments.txt

RUN python train.py
