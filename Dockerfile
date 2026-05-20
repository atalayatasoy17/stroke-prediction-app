FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /root/.kaggle
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY
RUN echo "{\"username\":\"${KAGGLE_USERNAME}\",\"key\":\"${KAGGLE_KEY}\"}" > /root/.kaggle/kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json

RUN python modeling/train.py

EXPOSE 8501

CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]