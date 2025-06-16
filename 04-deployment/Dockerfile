FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "requirements.txt", "./" ]

RUN pipenv install --system --deploy

COPY [ "main.py", "./" ]

ENTRYPOINT ["python", "main.py"]
