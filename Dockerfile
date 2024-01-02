FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

ENV PORT=8501

EXPOSE ${PORT}

CMD [ "streamlit", "run", "app.py" ]
