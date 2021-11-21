name="pytorch_21.10-py3_baseline"
docker login gitlab-master.nvidia.com:5005
docker build -t gitlab-master.nvidia.com:5005/ksivamani/containers:"$name" .
docker run -it --runtime=nvidia --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 -v /mnt/nvdl/usr/charleney/ds/ThePile:/data/path -v /mnt/nvdl/usr/ksivamani/data/gpt2:/files --ipc=host gitlab-master.nvidia.com:5005/ksivamani/containers:"$name"
