FROM python:3.6

WORKDIR /www

ADD . .

RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

EXPOSE 5000/tcp

CMD ["python", "run.py"]