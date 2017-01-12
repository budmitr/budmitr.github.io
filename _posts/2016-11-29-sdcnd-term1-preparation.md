---
layout: post
title:  "SDCND Term1 Preparations"
date:   2016-11-29 14:00:00 +0000
categories:
---

Recently I joined Udacity [Self-Driving Car Engineer Nanodegree](https://www.udacity.com/drive), which is a 9 month educational program consisting of 3 terms of 3 months each.
This post is not for detailed description of the course and its content, this can be found on Udacity homepage itself.  
Just in few words, first term of the program covers two main topics: Deep Learning and Computer Vision.
Programming exercises and projects for this term rely on python stack (tensorflow, keras, opencv).
To make assignments, one can have localhost machine setup or use AWS, use different operational systems etc.
Further terms will include more software and more specific requirements.
All of this give us next problem of environment setup.

## Problem

Prepare python-based working environment for term 1 with next requirements:
* agility -- solution should be runnable on different hosts;
* speed -- once created, enviroment should be easily repeated (on same host or on others);
* isolation -- all required software with its specific version should be isolated;
* GPU-ready -- for deep learning tasks the solution must be ready to use GPU.

## Approach

Solution approach is to use [Docker](https://www.docker.com/) which makes isolated **containers** based on **images**.
First questions I asked myself (and google then) were *'Hey, is this possible to use GPU with docker? If yes, how much the performance suffers?'*.

Spoiler! The answer is *'yes, it is possible and performance is good'*.

## Solution

I created 2 Docker files with environment description: one for CPU and one for GPU setup.
Both use [Anaconda](https://www.continuum.io/) for python stack environment management.

Here is the code of CPU-environment, the explanation is below and the up-to-date code is located at
{% include icon-github.html username="budmitr" %} /
[environments](https://github.com/budmitr/environments/tree/master/sdcnd-term1-cpu)

{% highlight docker %}
FROM ubuntu:xenial

MAINTAINER Dmitrii Budylskii

# Configure environment
ENV CONDA_DIR /opt/miniconda
ENV USER dockeruser
ENV USERGROUP 1000
ENV HOME /home/$USER

# Basic setups
EXPOSE 8888
RUN apt-get -y update && \
    apt-get install -y wget python-pip python-dev libgtk2.0-0 unzip

# Create local user
RUN useradd -m -s /bin/bash -N -u $USERGROUP $USER

# Install conda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -q -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    chown $USER $CONDA_DIR -R

# Switch to local user
USER $USER
ENV PATH $CONDA_DIR/bin:$PATH
WORKDIR $HOME

# prepare default python 3.5 environment
RUN pip install --upgrade pip && \
    pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0-cp35-cp35m-linux_x86_64.whl && \
    pip install h5py jupyter keras matplotlib moviepy pandas pillow sklearn flask-socketio eventlet && \
    conda install -y -c menpo opencv3=3.1.0 && \
    conda install -y seaborn
{% endhighlight %}

## Explanation

Lets go line by line and understand wtf is going on here.

{% highlight docker %}
FROM ubuntu:xenial
MAINTAINER Dmitrii Budylskii
{% endhighlight %}

CPU-environment is based on Ubuntu 16.04 LTS image which is xenial.
First line says that first of all we have to take predefined Docker image with installed Ubuntu and work with it.
Second line just tells who created this awesome Dockerfile.

{% highlight docker %}
# Configure environment
ENV CONDA_DIR /opt/miniconda
ENV USER dockeruser
ENV USERGROUP 1000
ENV HOME /home/$USER
{% endhighlight %}

These lines just set some environment variables so we can refer them later in the code. Easy...

{% highlight docker %}
# Basic setups
EXPOSE 8888
RUN apt-get -y update && \
    apt-get install -y wget python-pip python-dev libgtk2.0-0 unzip
{% endhighlight %}

OK this is more interesting part.
We say that port 8888 which is used by jupyter notebook is exposed to host machine.
This allows us to connect to jupyter being outside of docker container, i.e. from our host or even remote machine.
Next two lines is a oneliner command which updates and installs to ubuntu listed packages.

After packages are installed to the system, I want to escape from root user and make a local one which is called here 'dockeruser'.

{% highlight docker %}
# Create local user
RUN useradd -m -s /bin/bash -N -u $USERGROUP $USER
{% endhighlight %}

At this moment we finished 'global installations' and now we want to install anaconda.
Next 4 lines of code download miniconda (same as anaconda but smaller) into temporaty file miniconda.sh and launches it.
Local user 'dockeruser' is going to be the owner of miniconda folder to be able to install new packages without sudo.

{% highlight docker %}
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -q -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    chown $USER $CONDA_DIR -R
{% endhighlight %}

Okay, whats next?
We have our package manager so its time to install python packages.
To do this under our local user, we first setup path to conda in environment to make `conda` command available.
After its done, we only need to install packages with `pip install` and `conda install`

{% highlight docker %}
# Switch to local user
USER $USER
ENV PATH $CONDA_DIR/bin:$PATH
WORKDIR $HOME

# prepare default python 3.5 environment
RUN pip install --upgrade pip && \
    pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0-cp35-cp35m-linux_x86_64.whl && \
    pip install h5py jupyter keras matplotlib moviepy pandas pillow sklearn flask-socketio eventlet && \
    conda install -y -c menpo opencv3=3.1.0 && \
    conda install -y seaborn
{% endhighlight %}

Lof of 'pip install' here make all the dirty stuff for us.
And this is actually it!
Dockerfile is ready to be built into docker image. Just use terminal command for it

`docker build -t sdcnd-term1-cpu environments/sdcnd-term1-cpu`

Note that this build does not have 'budmitr' prefix, which I use in github version.

## Usage

Suppose you have 'sdcnd' folder in your home directory where you put projects.
The last step is to run

`docker run -d -p 8888:8888 -u dockeruser -v ~/sdcnd:/home/dockeruser/ budmitr/sdcnd-term1-cpu sh -c "jupyter notebook --ip=* --no-browser"`

* `-d` runs the docker in detached container, so your terminal is free for usage. You can find this container with `docker ps` command, stop it with `docker stop <container_id>`, start again with `docker start <container_id>` and even remove this container with `docker rm <container_id>`
* `-p 8888:8888` says that we want to use exposed port 8888 from docker at the same port 8888 in host machine
* `-v ~/sdcnd:/home/dockeruser/` this part allowes you to use your project files under jupyter notebook since folder will be linked to working directory of docker user
* `budmitr/sdcnd-term1-cpu` is image name which you can also run without build, because its in docker hub
* `sh -c "jupyter notebook --ip=* --no-browser"` launches jupyter notebook. **NOTE:** we have to use `sh -c` because jupyter kernels are broken under docker without it

That's it!
Just go to `http://localhost:8888` and start doing your projects!

## GPU-ready version

Dockerfile for GPU version is almost same and you can get it here:
{% include icon-github.html username="budmitr" %} /
[environments](https://github.com/budmitr/environments/tree/master/sdcnd-term1-gpu)

You need to have GPU drivers installed in your host system already!

What is the difference?
There are only three.

First is to use nvidia/cuda:cudnn image instead of just ubuntu:xenial
`FROM nvidia/cuda:cudnn`

Second difference is to use GPU-version of tensorflow

Third and main difference is that you have to use `nvidia-docker` instead of `docker`.
See installation details here: (https://github.com/NVIDIA/nvidia-docker).

So the running command changes to this one:
`nvidia-docker run -d -p 8888:8888 -u dockeruser -v ~/sdcnd:/home/dockeruser/ budmitr/sdcnd-term1-gpu sh -c "jupyter notebook --ip=* --no-browser"`

## Small benchmark

As I noted in Slack chat of the Nanodegree program, guys use tensorflow convolution script and check how fast it is.

{% highlight shell %}
wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/models/image/mnist/convolutional.py
python convolutional.py
{% endhighlight %}

As you can see below, docker and non-docker version have same performance with minor deviations of about 1ms.

### GPU results (TITANX Maxwell)

{% highlight shell %}
$ python convolutional.py
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcurand.so locally
Extracting data/train-images-idx3-ubyte.gz
Extracting data/train-labels-idx1-ubyte.gz
Extracting data/t10k-images-idx3-ubyte.gz
Extracting data/t10k-labels-idx1-ubyte.gz
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:936] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties:
name: GeForce GTX TITAN X
major: 5 minor: 2 memoryClockRate (GHz) 1.076
pciBusID 0000:01:00.0
Total memory: 11.94GiB
Free memory: 11.76GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX TITAN X, pci bus id: 0000:01:00.0)
Initialized!
...
Step 0 (epoch 0.00), 11.5 ms
...
Step 100 (epoch 0.12), 5.7 ms
...
Step 200 (epoch 0.23), 5.5 ms
...
Step 300 (epoch 0.35), 5.2 ms
...
Step 400 (epoch 0.47), 5.4 ms
...
{% endhighlight %}

### Docker GPU results (TITANX Maxwell)

{% highlight shell %}
$ python convolutional.py
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcurand.so locally
Extracting data/train-images-idx3-ubyte.gz
Extracting data/train-labels-idx1-ubyte.gz
Extracting data/t10k-images-idx3-ubyte.gz
Extracting data/t10k-labels-idx1-ubyte.gz
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:936] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties:
name: GeForce GTX TITAN X
major: 5 minor: 2 memoryClockRate (GHz) 1.076
pciBusID 0000:01:00.0
Total memory: 11.94GiB
Free memory: 11.76GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX TITAN X, pci bus id: 0000:01:00.0)
Initialized!
Step 0 (epoch 0.00), 8.1 ms
...
Step 100 (epoch 0.12), 5.7 ms
...
Step 200 (epoch 0.23), 5.4 ms
...
Step 300 (epoch 0.35), 5.4 ms
...
Step 400 (epoch 0.47), 5.4 ms
...
{% endhighlight %}
