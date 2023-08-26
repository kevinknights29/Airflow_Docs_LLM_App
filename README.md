# Airflow_Docs_LLM_App

This is an application built with LLMs to enable Airflow documentation Q&amp;A, with the intent of facilitating the process of building Airflow DAGs.

This project uses CodeLlama Python 7B model to power the code inference.

You can learn more about Llama code models at: [Code Llama](https://ai.meta.com/blog/code-llama-large-language-model-coding/)

## Current Progress

![image](https://github.com/kevinknights29/Flask-REST-API/assets/74464814/f534ee23-5a28-465d-875c-96d81339c74d)

## Usage

This project is build around Docker.

To get started, simply:

- Build the docker images from the different services.

```bash
docker compose build
```

- Start the docker container services required, and have fun.

```bash
docker compose up -d
```

- To shut the services down, run the following:

```bash
docker compose down
```

## Contributing

### Installing pre-commit

Pre-commit is already part of this project dependencies.
If you would like to installed it as standalone run:

```bash
pip install pre-commit
```

To activate pre-commit run the following commands:

- Install Git hooks:

```bash
pre-commit install
```

- Update current hooks:

```bash
pre-commit autoupdate
```

To test your installation of pre-commit run:

```bash
pre-commit run --all-files
```
