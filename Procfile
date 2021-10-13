web: gunicorn app:app
docker pull tensorflow/serving
worker: docker run -p 8501:8501 --name tfs --mount type=bind,source=currentModel,target=/models/model -t tensorflow/serving