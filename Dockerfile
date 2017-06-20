FROM gcr.io/cloud-datalab/datalab:latest

RUN pip install -U --no-cache-dir apache_beam google-cloud-dataflow

