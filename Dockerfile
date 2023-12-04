FROM ubuntu

USER root

# Install Python 3.11 and pip
RUN apt-get update && apt-get install -y python3.11 python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python

# Update alternatives to use Python 3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install Java 17 Amazon Corretto
RUN apt-get install -y wget gnupg && \
    wget https://corretto.aws/downloads/latest/amazon-corretto-17-x64-linux-jdk.tar.gz && \
    tar -xzf amazon-corretto-17-x64-linux-jdk.tar.gz -C /usr/local && \
    rm amazon-corretto-17-x64-linux-jdk.tar.gz && \
    mv /usr/local/amazon-corretto-17.0.9.8.1-linux-x64 /usr/local/amazon-corretto-17
ENV JAVA_HOME=/usr/local/amazon-corretto-17
ENV PATH=$PATH:$JAVA_HOME/bin

# Set working directory
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code and required files
COPY src /app/src
COPY checkstyle /app/checkstyle
COPY scripts /app/scripts

# Give write permission to the user
RUN chmod -R 777 /app

# Set Python path
ENV PYTHONPATH /app:/app/src:/app/src/styler2_0:/app/scripts:$PYTHONPATH
