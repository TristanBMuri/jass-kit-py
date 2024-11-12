FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt /app/

RUN pip install -r requirements.txt


COPY jass /app/jass
COPY examples/serverless/app.py /app/

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# Expose port 8888 for Flask
EXPOSE 8888

# Set the default command to activate the environment and run your script
CMD ["flask", "run", "--host=0.0.0.0", "--port=8888"]