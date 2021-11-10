name="bf16.gpt3.126m.linear.e4m3.f1.p0"
docker login gitlab-master.nvidia.com:5005
docker build -t "$name" .
docker run -it --rm --runtime=nvidia --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 -v /mnt/nvdl/usr/charleney/ds/ThePile:/data/path -v /mnt/nvdl/usr/ksivamani/data/gpt2:/files --ipc=host "$name"
