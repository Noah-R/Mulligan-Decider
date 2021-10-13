FROM tensorflow/serving
RUN -p 8501:8501 --name tfs --mount type=bind,source=currentModel,target=/models/model