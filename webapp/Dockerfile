FROM python:3.7-alpine
RUN mkdir /app
WORKDIR /app
COPY * /app
RUN pip3 install -r requirements.txt
CMD gunicorn app:app