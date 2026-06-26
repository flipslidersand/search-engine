FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY searchengine/ ./searchengine/

ENV SEARCH_DB=/data/search.db

EXPOSE 8000

CMD ["uvicorn", "searchengine.server:app", "--host", "0.0.0.0", "--port", "8000"]
