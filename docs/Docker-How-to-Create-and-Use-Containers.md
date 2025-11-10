# 🐳 Docker: How to Create and Use Containers

Docker provides a lightweight way to package and run applications inside isolated environments called **containers**.  
A container runs from a **Docker image**, which is built from a **Dockerfile** — a step-by-step recipe that defines how the environment should be set up.

---

## 1. Example: `Dockerfile`

```python
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
  
# apt packages
RUN apt-get update && apt-get install curl wget git pip vim tmux htop libxml2 kmod systemctl lsof python3.10 -y
RUN pip install --upgrade pip
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /root

# Python library
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip install transformers accelerate sentencepiece tokenizers texttable toml attributedict protobuf cchardet
RUN pip install matplotlib scikit-learn pandas

# vimrc               
RUN echo "abbr set_pdb import pdb; pdb.set_trace()" >> ~/.vimrc
RUN echo "abbr cmt #============================ <Esc>i" >> ~/.vimrc

# Jupyter notebook
RUN pip install jupyter
RUN jupyter notebook --generate-config
RUN echo "c.ServerApp.allow_remote_access = True" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "jupyter notebook --ip 0.0.0.0 --allow-root --port=9000 --no-browser &" >> /root/jupyter-run.sh
RUN echo "lsof -nP -iTCP:9000 | grep LISTEN" >> /root/check-port.sh

# Flush
RUN rm -rf /root/.cache/pip
```

---

## 2. How to Create a Dockerfile

1. Inside your working directory, create a file named **`Dockerfile`** (no extension).  
   Each command in this file is executed **line-by-line** in order.

2. Start from a **base image**.  
   You can build your environment from Ubuntu manually, but it is much easier to use NVIDIA’s official CUDA base images.  
   → Available at: [https://hub.docker.com/r/nvidia/cuda/tags](https://hub.docker.com/r/nvidia/cuda/tags)

3. Use the `FROM` command to specify the base image.  
   Example:  
   ```dockerfile
   FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
   ```

4. To execute shell commands (e.g., `apt-get install`, `pip install`, `ln -s`), use the `RUN` directive.  
   This is where you install Linux packages and Python libraries.

5. The `WORKDIR` command sets the working directory for all subsequent instructions.

6. For configurations (e.g., `.vimrc` settings), you can use `echo` and the `>>` operator to append text directly.

---

## 3. Building a Docker Image

Once the `Dockerfile` is ready, build the image using:

```bash
docker build . --tag mydocker:1.0
```

- `--tag` (or `-t`) assigns a name and version tag to the image (e.g., `mydocker:1.0`).  
- The final `.` specifies the **build context**, meaning Docker will look for the Dockerfile in the current directory.  
- You can view all your built images with:
  ```bash
  docker images
  ```

---

## 4. Running a Docker Container

Now that the image is built, you can launch a container interactively:

```bash
docker run -it --rm --gpus all     -p 9012:9000     --ipc=host     -v /home/user/workspace/mx-lrm:/root/mx-lrm     -v /raid:/raid     mydocker:1.0 bash
```

**Explanation of options:**

- `-it` → Interactive terminal mode (so you can type commands inside the container).  
- `--rm` → Automatically removes the container when it exits, preventing leftover containers.  
- `--gpus all` → Exposes all available GPUs to the container (you can specify indices if needed).  
- `-p 9012:9000` → Port forwarding. Accessing `localhost:9012` on the host corresponds to port `9000` inside the container.  
- `--ipc=host` → Shares the host’s inter-process communication (important for PyTorch DDP training).  
- `-v [host_path]:[container_path]` → Mounts a local directory into the container. Changes are reflected both ways.

Example:
- `/home/user/workspace/mx-lrm` → Mounted at `/root/mx-lrm`  
- `/raid` → Mounted directly at `/raid`

**Tip:**  
The container runs in an isolated environment — any changes not saved in a mounted directory will be lost once the container stops.  
Therefore, mounting key directories with `-v` is strongly recommended.

---

## 4.1 Saving Your Container as a New Image (`docker commit`)

Sometimes, after launching a container, you may install additional packages, change configurations, or update code inside it.  
If you want to **save the current container state as a new reusable image**, use the `docker commit` command.

```bash
docker ps
```
→ Find the **CONTAINER ID** of the running container you want to save.

Then run:
```bash
docker commit [CONTAINER_ID] mydocker:updated
```

**Explanation:**
- `docker commit` creates a new image from your current container.
- `mydocker:updated` is the name and tag of the new image.
- After this, you can run a container directly from this updated image:
  ```bash
  docker run -it --rm --gpus all -p 9012:9000 --ipc=host -v /home/user/workspace/mx-lrm:/root/mx-lrm -v /raid:/raid mydocker:updated bash
  ```

---

## 5. Container Management

- List all running containers:
  ```bash
  docker ps
  ```
- List all containers (including stopped ones):
  ```bash
  docker ps -a
  ```
- Stop a running container:
  ```bash
  docker stop [CONTAINER_ID]
  ```
- Remove a container manually:
  ```bash
  docker rm [CONTAINER_ID]
  ```
- View all images:
  ```bash
  docker images
  ```
- Remove unused images:
  ```bash
  docker image prune
  ```

---

## ✅ Summary

| Concept | Description |
|----------|--------------|
| **Dockerfile** | A script defining how to build your image (base image, dependencies, setup). |
| **Image** | A snapshot of an environment that can be instantiated. |
| **Container** | A running instance of an image (temporary by nature). |
| **Mount (-v)** | Keeps your code and results persistent by linking local and container directories. |
| **Port (-p)** | Allows network access to services running inside the container. |
 
