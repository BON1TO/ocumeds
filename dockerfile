FROM python:3.11

WORKDIR /code

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

EXPOSE 80

CMD ["uvicorn", "ocumed_api:app", "--host", "0.0.0.0", "--port", "80", "--reload", "--workers", "1"]