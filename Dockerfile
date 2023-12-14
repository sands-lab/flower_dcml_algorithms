FROM python:3.8-slim-bullseye

RUN apt update
RUN apt install build-essential -y
RUN pip install --upgrade pip

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["/bin/bash", "./run_experiment.sh"]
