FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt /app/

RUN pip install -r requirements.txt


COPY test1/flask_app.py /app/


# Set the default command to activate the environment and run your script
CMD ["python", "flask_app.py"]
