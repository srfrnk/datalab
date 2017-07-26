FROM gcr.io/cloud-datalab/datalab:latest

RUN apt-get update && \
    apt-get -y -o Dpkg::Options::="--force-confnew" upgrade && \
    apt-get -y -o Dpkg::Options::="--force-confnew" dist-upgrade && \
    apt-get -y autoremove && \
    apt-get -y autoclean

COPY requirements.txt /requirements.txt

RUN pip install -U --no-cache-dir -r requirements.txt
