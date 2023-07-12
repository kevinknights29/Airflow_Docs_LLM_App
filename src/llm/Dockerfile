FROM jupyter/scipy-notebook:latest
LABEL authors="Kevin Knights | kevinknights29"

USER root

RUN python -m pip install --upgrade pip

RUN pip install jupyterlab-code-formatter --user
RUN pip install jupyterlab-lsp 'python-lsp-server[all]' --user
RUN pip install jupyterlab-spellchecker --user

ENV JUPYTER_ENABLE_LAB=yes
ENV CHOWN_HOME=yes
