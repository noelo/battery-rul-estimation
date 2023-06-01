# FROM quay.io/opendatahub/workbench-images:cuda-ubi8-python-3.8-pr-94
# FROM quay.io/opendatahub/workbench-images:runtime-cuda-tensorflow-ubi8-python-3.8
FROM docker.io/tensorflow/tensorflow:latest-gpu
USER 0
# RUN yum install -y tensorrt \
#   python3-libnvinfer=8.6.0.12-1+cuda11.8 \
#   libnvinfer8=8.6.0.12-1+cuda11.8 \
#   libnvinfer-plugin8=8.6.0.12-1+cuda11.8 \
#   libnvinfer-vc-plugin8=8.6.0.12-1+cuda11.8 \
#   libnvparsers8=8.6.0.12-1+cuda11.8 \
#   libnvonnxparsers8=8.6.0.12-1+cuda11.8 \
#   libcudnn8=8.8.1.3-1+cuda11.8 \
#     && yum clean all \
#     && rm -rf /var/cache/yum/*

#Add application sources with correct permissions for OpenShift
# USER 0

ARG UID=1001
ARG GID=1001

RUN groupadd -g "${GID}" battery \
  && useradd --create-home --no-log-init -u "${UID}" -g "${GID}" battery

RUN mkdir -p /opt/app-root/src/
# ADD app-src .
RUN chown -R 1001:1001 /opt/app-root/src/
WORKDIR /opt/app-root/src/
# USER 1001
COPY requirements.txt .
RUN pip install -r requirements.txt 
RUN pip list

USER 1001
ADD battery-rul-estimation/data_processing/ data_processing

RUN cd /opt/app-root/src/

# Install the dependencies


ENV PYTHONPATH=/opt/app-root/src/
USER 1001