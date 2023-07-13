FROM jupyter/scipy-notebook:latest
LABEL authors="Kevin Knights | kevinknights29"

USER root

RUN python -m pip install --upgrade pip

ENV JUPYTER_ENABLE_LAB=yes

# Adding theme and extensions
COPY ./overrides.json /opt/conda/share/jupyter/lab/settings/overrides.json
