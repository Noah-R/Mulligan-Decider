FROM python:3.7-alpine
RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app
COPY app.py /app
COPY prediction.py /app
RUN pip3 install -r requirements.txt
CMD gunicorn app:app