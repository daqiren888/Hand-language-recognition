import jetson.inference
import jetson.utils
import paho.mqtt.client as mqtt
import json
##LOCAL_MQTT_HOST="mosquitto-service"
LOCAL_MQTT_HOST="10.43.177.80"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="my_image"
def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker on Jetson with rc: " + str(rc))
local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
net = jetson.inference.detectNet(argv=["--model=/jetson-inference/python/training/detection/ssd/models/W251/ssd-mobilenet.onnx","--labels=/jetson-inference/python/training/detection/ssd/models/W251/labels.txt","--input-blob=input_0","--output-cvg=scores","--output-bbox=boxes"],threshold=0.5)
camera = jetson.utils.videoSource("/dev/video0")
display = jetson.utils.videoOutput("display://0")
dict_keys = ['frame_number', 'object', 'location', 'det_confidence']
detection_results = dict.fromkeys(dict_keys)
img_counter = 0
while display.IsStreaming():
    img = camera.Capture()
    detections = net.Detect(img)
    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
    if (img_counter % 30) == 0:  # Send detection meta-data every 30 frames
        for detect in detections:  # Loop through all detected objects in a frame
            class_id = detect.ClassID  # Get the Class Id of the object
            # Prepase the MQTT message
            detection_results['frame_number'] = img_counter
            detection_results['object'] = net.GetClassDesc(class_id)  # Get the object name
            # Get the (x,y) coordinates of the object
            detection_results['location'] = detect.Center
            # Confidence value of the detection
            detection_results['det_confidence'] = detect.Confidence
            # Publish the MQTT message
            msg = json.dumps(detection_results)
            local_mqttclient.publish(LOCAL_MQTT_TOPIC, payload=msg, qos=0, retain=False)
    display.Render(img)  # Render out this frame with detections
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
local_mqttclient.loop_forever()
