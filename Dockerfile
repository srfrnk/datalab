FROM gcr.io/cloud-datalab/datalab:latest

RUN apt-get update && \
    apt-get -y -o Dpkg::Options::="--force-confnew" upgrade && \
    apt-get -y -o Dpkg::Options::="--force-confnew" dist-upgrade && \
    apt-get -y autoremove && \
    apt-get -y autoclean

RUN pip install -U --no-cache-dir google-cloud-dataflow oauth2client ipykernel
    # apache-beam apache-beam[gcp,test,docs] \
    # google-api-python-client google-cloud-dataflow tensorflow-transform tensorflow seaborn \
    # plotly numpy pandas scipy scikit-learn sympy statsmodels tornado pyzmq jinja2 jsonschema \
    # python-dateutil pytz pandocfilters pygments argparse mock requests oauth2client httplib2 \
    # futures PyYAML six ipykernel future nltk bs4 crcmod pillow protobuf==3.1.0