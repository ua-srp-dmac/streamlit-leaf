FROM python:3.9

RUN apt-get update && apt-get install -y \
	python3-opencv python3-dev zbar-tools libzbar0

RUN apt-get install default-jdk -y

RUN pip install torch==1.10.2 torchvision==0.11.3 -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip install 'git+https://github.com/facebookresearch/fvcore'

# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
RUN pip install -e detectron2_repo

# install other requirements
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY entry.sh /bin

COPY . /app

EXPOSE 8501

ENTRYPOINT ["bash", "/bin/entry.sh"]

