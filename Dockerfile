FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir pandas scikit-learn

CMD ["python", "spam_detector.py"]
