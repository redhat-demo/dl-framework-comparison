# Use the official Python image from the Docker Hub
FROM nvcr.io/nvidia/pytorch:24.04-py3

COPY requirements_torch.txt /tmp/
RUN pip install --requirement /tmp/requirements_torch.txt
# Specify the command to run Jupyter Notebook within the container
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
