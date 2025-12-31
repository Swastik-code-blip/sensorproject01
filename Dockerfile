FROM py thon:3.8-slim-buster

WORKDIR /app

COPY . /app/

CMD ["python3","app.py"]