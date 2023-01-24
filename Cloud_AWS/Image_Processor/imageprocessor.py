import paho.mqtt.client as mqtt
import sys
import uuid
import boto3

# If Local Host does not work, use the service IP directly
LOCAL_MQTT_HOST="mosquitto-service"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="mosquitto-face"

# Populate the aws_access_key_id and aws_secret_access_key with your users
S3_client = boto3.client('s3',
                         aws_access_key_id="",
                         aws_secret_access_key="")

def save_image(img_bytes):
    S3_client.put_object(
        Bucket = 'w251-hw3-bucket-jm',
        Body= img_bytes,
        Key = str(uuid.uuid4()),
        ContentType='image/png'
    )

def on_connect(client, userdata, flags, rc):
    print("connected to local broker with rc: " + str(rc))
    client.subscribe(LOCAL_MQTT_TOPIC)

def on_message(client,userdata, msg):
    print("in the try message")
    save_image(msg.payload)
    print("saved image!")

local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.on_message = on_message

local_mqttclient.loop_forever()
