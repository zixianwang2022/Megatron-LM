ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.10-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace/
ADD . /workspace/gpt3
WORKDIR /workspace/gpt3
