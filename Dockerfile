FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

ENV PYTHON_VERSION 3.7.1
ENV HOME /root
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT $HOME/.pyenv


RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install tzdata
RUN apt-get -y install git make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev && \
    apt-get -y install wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev && \
    git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT && \
    $PYENV_ROOT/plugins/python-build/install.sh && \
    /usr/local/bin/python-build -v $PYTHON_VERSION $PYTHON_ROOT && \
    rm -rf $PYENV_ROOT

RUN pip install -y jupyter 
   

RUN pip install -U pip && \
    pip install fastprogress japanize-matplotlib

RUN apt-get install -y mecab && \
    apt-get install -y libmecab-dev && \
    apt-get install -y mecab-ipadic-utf8 && \
    apt-get install -y git && \
    apt-get install -y make && \
    apt-get install -y curl && \
    apt-get install xz-utils && \ 
    apt-get install file && \
    apt-get install sudo && \
    pip install -r requirements.txt

RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git && \
    cd mecab-ipadic-neologd && \
    bin/install-mecab-ipadic-neologd -n -y

RUN pip insatall mecab-python3

RUN curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash - \
    && sudo apt-get install -y nodejs

# キャッシュを消してイメージを小さくする
RUN apt-get clean -y && \
    apt-get autoremove -y && \
    apt-get update -y && \
    apt-get upgrade -y && \
    apt-get autoremove -y && \
    apt-get autoclean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN useradd -m -s /bin/bash vov
USER vov

RUN jupyter notebook --generate-config && \
    sed -i.back \
    -e "s:^#c.NotebookApp.token = .*$:c.NotebookApp.token = u'':" \
    -e "s:^#c.NotebookApp.ip = .*$:c.NotebookApp.ip = '0.0.0.0':" \
    -e "s:^#c.NotebookApp.open_browser = .*$:c.NotebookApp.open_browser = False:" \
    /home/${USERNAME}/.jupyter/jupyter_notebook_config.py

WORKDIR /home/vov
