FROM python:3.12-alpine

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

ENV LLM_PROXY_SERVICE_API_URL=https://codemie.lab.epam.com/llms
ENV LLM_PROXY_SERVICE_API_KEY=sk-***********************

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]