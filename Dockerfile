FROM registry.access.redhat.com/ubi8/python-38

# Add application sources with correct permissions for OpenShift
# USER 0
# ADD app-src .
# RUN chown -R 1001:0 ./
# USER 1001
COPY requirements.txt .
COPY battery-rul-estimation/data_processing/ .

# Install the dependencies
RUN pip install -r requirements.txt 
ENV PYTHONPATH=/opt/app-root/src