FROM quay.io/domino/python-public:3.8.7-slim
RUN apt-get update && apt-get upgrade -y
ADD requirements.txt .
ENV PATH=$PATH:/app/.local/bin:/app/bin
ENV PYTHONUNBUFFERED=true
ENV PYTHONUSERBASE=/app
ENV FLASK_ENV=production
ENV LOG_LEVEL=WARNING
RUN pip install --upgrade pip
RUN pip install --user -r requirements.txt
ADD domino-extensions-api /app
ENTRYPOINT ["python",  "/app/api.py", "http://nucleus-frontend.domino-platform:80","/"]
