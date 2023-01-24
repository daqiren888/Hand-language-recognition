import sys
import cv2 as cv
import paho.mqtt.client as mqtt

LOCAL_MQTT_HOST="mosquitto-service"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="mosquitto-face"

def on_connect_local(client, userdata, flags, rc):
    print("connected to local broker with rc: " + str(rc))
    client.subscribe(LOCAL_MQTT_TOPIC)

local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.loop_start()

cam = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

while(True):
    ret, frame = cam.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for face_details in faces:
        print("new face")
        (x, y, w, h) = face_details
        face = gray[y: y + h, x: x + w]
        _, pic = cv.imencode('.png', face)
        local_mqttclient.publish(LOCAL_MQTT_TOPIC, payload=pic.tobytes(), qos=0, retain=False)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
