# Build from the REPO ROOT:   docker build -f ml-service/Dockerfile -t saanscare-ml .
FROM python:3.11-slim
WORKDIR /app
COPY ml-service/requirements.txt ml-service/requirements.txt
RUN pip install --no-cache-dir -r ml-service/requirements.txt
COPY ml-service ml-service
COPY models models
WORKDIR /app/ml-service
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
