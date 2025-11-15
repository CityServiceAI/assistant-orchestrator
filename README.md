### Install dependencies
```shell
pip install -r requirements.txt
```

### Run application
```shell
fastapi dev app/main.py
```

### Build docker image
```shell
docker build -t assistant-orchestrator:1 .
```

### OpenAPI doc
Open http://127.0.0.1:8000/docs page