FROM tensorflow/tensorflow:latest

COPY app_streamlit_docker.py /app_streamlit_docker.py
COPY project_watson/data/train.csv /project_watson/data/train.csv
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT [ "streamlit", "run", "app_streamlit_docker.py" ]