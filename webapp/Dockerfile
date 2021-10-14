FROM python:3
RUN mkdir /app
WORKDIR /app
COPY . /app/
RUN pip3 install -r requirements.txt
ENV PORT 8000
EXPOSE $PORT
CMD gunicorn --bind 0.0.0.0:$PORT app:app