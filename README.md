# W251-Final-Project
Instructors: Ryan DeJana, Prabhakar Attaluri

Author: Johnathan Mah, Rohit Bakshi, Da Qi Ren, Sitao Chen

## Credit
A large portion of this repository uses the jetson-inference repository by NVIDIA -
https://github.com/dusty-nv/jetson-inference as reference. Several of the python
scripts were modified to adjust to our particular use case. 

## Introduction
This repository holds the code to run a mobilenet_v2 model that is trained to detect
American Sign Language. The model can be trained by navigating to the following
path and running the train_ssd.py script. Just make sure you have training data
setup under the folder structure with VOC xml. After training you can export the
pytorch model to an onnx model to run TensorRT.
```
/python/training/detection/ssd
```

## Prerequisites
Before attempting to run the application end to end please be aware of the
following.

1. Configure an AWS Cloud Instance.
2. Configure a public AWS S3 Bucket.
3. Make sure you have permissions to put objects into the S3 Bucket.
4. Keep your AWS Access Key and AWS Secret Access Key handy. You will need to
pass them in the imageprocessor.py file to store images onto the S3 Bucket.
5. You will need k3s (Kubernetes) on both the Jetson Nano and the AWS Cloud
Instance.
6. Have a Jetson Device handy.

## Starting the Model Detection on the Jetson
To get a model trained and running run the following commands:
```
python3 train_ssd.py --dataset-type=voc --data=data/<name> --model-dir=models/<name> --batch-size=# --workers=# --epochs=#

python3 onnx_export.py --model-dir=models/<name>

detectnet --model=models/<name>/ssd-mobilenet.onnx --labels=models/W251/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes /dev/video0
```

If you do not want to run on a live video feed, replace /dev/video0 with a
folder path that contains a set of images to run tests on.

## How to Run the Application w/ MQTT
Make sure you have Kubernetes up and running on both the Jetson Nano and Cloud
Instance.

We will be building Docker containers and initializing them in Kubernetes. The
files are neatly bundled under their respective folders - Jetson or Cloud_AWS.

Note: The commands below are referencing my Docker Hub Account, you can build,
tag, and push to your own account. Just change the yaml files accordingly.

Note: In order for the application to run be sure you pass in your AWS Keys in
detectMQTT.py

Note: The exposed ports used in the files need to be updated to your instances ports.

Run the following:
```
python3 detectMQTT.py
```
This will launch the detection model and for every detection we will send information to the cloud (s3 bucket). 

### Jetson
1. Launch the Local MQTT Broker (Jetson/Jetson_Broker)
```
docker build --no-cache -t johnymah/mosquitto:v1 .
docker push johnymah/mosquitto:v1
kubectl apply -f mosquitto.yaml
kubectl apply -f mosquittoService.yaml
```

2. Launch the Face Detector (Jetson/Face_Detector)
```
docker build --no-cache -t johnymah/facedetector:v1 .
docker push johnymah/facedetector:v1
kubectl apply -f facedetector.yaml
```

3. Launch the MQTT Message Forwarder (Jetson/Forwarder)
```
docker build --no-cache -t johnymah/forwarder:v1 .
docker push johnymah/forwarder:v1
kubectl apply -f forwarder.yaml
```

### Cloud
1. Launch the Cloud MQTT Broker
```
docker build --no-cache -t johnymah/mosquitto:v1 .
docker push johnymah/mosquitto:v1
kubectl apply -f mosquitto.yaml
kubectl apply -f mosquittoService.yaml
```

2. Launch the Image Processor
```
docker build --no-cache -t johnymah/imageprocessor:v1 .
docker push johnymah/imageprocessor:v1
kubectl apply -f imageprocessor.yaml
```

### Shutdown
1. To Shutdown (Jetson)
```
kubectl delete deploy moquitto-forwarder-deployment
kubectl delete deploy facedetector
kubectl delete deploy mosquitto-deployment
kubectl delete service mosquitto-service
```
2. To Shutdown (Cloud)
```
kubectl delete deploy mosquitto-deployment
kubectl delete deploy image-processor
kubectl delete service mosquitto-service
```

### Useful Commands for Debugging / See Logs
1. To See Services or Deployments
```
kubectl get services
kubectl get deployments
```
2. To See Logs of Pods
```
kubectl get pods -l app=<insert app name>
kubectl get logs -f <pod name>
```
