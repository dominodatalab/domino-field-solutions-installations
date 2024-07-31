FROM quay.io/domino/python-slim:3.9.16-slim-bullseye-356299
ENV PATH=$PATH:/home/app/.local/bin:/home/app/bin
ENV PYTHONUNBUFFERED=true
ENV PYTHONUSERBASE=/home/app
ENV FLASK_ENV=production
ENV LOG_LEVEL=INFO
ENV MODEL_NAME=RandomForestRegression_2
ENV MODEL_VERSION=1

RUN groupadd --gid 1000 domino && \
    useradd --uid 1000 --gid 1000 domino -m -d /app

RUN apt-get update \
    && apt-get upgrade --yes \
    && apt-get install -y --no-install-suggests --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ADD artifacts model
RUN pip install --upgrade pip
RUN chmod -R 755 /app/model
RUN pip install -r  /app/model/sklearn-model/requirements.txt
COPY ./execute_model.py /app/model

USER 1000
ENTRYPOINT ["python"]
CMD ["model/execute_model.py"]
