# Use an official Python runtime as a parent image (change 'buster' to 'slim', if needed)
FROM python:3.11-buster  

LABEL maintainer="philip584521@gmail.com"  

WORKDIR /usr/src/app

COPY requirements.txt ./  
RUN pip install --no-cache-dir -r requirements.txt  

COPY . .

EXPOSE 8501  

CMD ["streamlit", "run", "app.py"]  