FROM tensorflow/serving
COPY currentModel models/model

ENV MODEL_BASE_PATH /models
ENV MODEL_NAME model

ENV PORT 8501
EXPOSE $PORT

COPY tf_serving_entrypoint.sh /usr/bin/tf_serving_entrypoint.sh
RUN chmod +x /usr/bin/tf_serving_entrypoint.sh
ENTRYPOINT []
CMD ["/usr/bin/tf_serving_entrypoint.sh"]