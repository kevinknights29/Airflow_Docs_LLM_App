# Sources:
# https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html
# https://jupyter-docker-stacks.readthedocs.io/en/latest/using/recipes.html#using-mamba-install-or-pip-install-in-a-child-docker-image
FROM jupyter/scipy-notebook:latest
LABEL authors="Kevin Knights | kevinknights29"

# Enable JupyterLab
ENV JUPYTER_ENABLE_LAB=yes

# Upgrade pip
RUN python -m pip install --upgrade pip

# Uncomment to install from requirements.txt file
# Install from the requirements.txt file
# COPY --chown=${NB_UID}:${NB_GID} requirements.txt /tmp/
# RUN pip install --no-cache-dir --requirement /tmp/requirements.txt && \
#     fix-permissions "${CONDA_DIR}" && \
#     fix-permissions "/home/${NB_USER}"

# Extensions for JupyterLab can be found: https://jupyterlab-contrib.github.io/migrate_from_classical.html
# Install JupyterLab LSP extension
RUN pip install --no-cache-dir jupyterlab-lsp 'python-lsp-server[all]' && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Install JupyterLab Code Formatter extension
RUN pip install --no-cache-dir jupyterlab-code-formatter black isort && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Install JupyterLab Execute Time extension
RUN pip install --no-cache-dir jupyterlab_execute_time && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Install JupyterLab Spell Checker extension
RUN pip install --no-cache-dir jupyterlab-spellchecker && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Adding theme and extensions
COPY ./overrides.json /opt/conda/share/jupyter/lab/settings/overrides.json
