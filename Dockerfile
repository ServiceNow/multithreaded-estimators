FROM python:3.6.2

WORKDIR /eai

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY ./threaded_estimator ./threaded_estimator
COPY ./setup.py ./setup.py
RUN pip install .

CMD bash -c "pytest -s --full-trace /eai/threaded_estimator/tests"
