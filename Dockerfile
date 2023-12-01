FROM python:3.11
USER root
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
COPY src /app/src
COPY checkstyle /app/checkstyle
COPY scripts /app/scripts
RUN chmod -R 777 /app
ENV PYTHONPATH /app:/app/src:/app/src/styler2_0:/app/scripts:$PYTHONPATH