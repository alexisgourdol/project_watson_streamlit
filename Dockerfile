FROM tensorflow/tensorflow:latest

WORKDIR /app

COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8501

COPY app.py /app.py

CMD streamlit run --server.port 8501 --server.enableCORS false app.py

ENTRYPOINT [ "streamlit", "run", "app.py" ]
