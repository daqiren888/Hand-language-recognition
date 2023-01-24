import paho.mqtt.client as mqtt
import sys

# Change Remote Host to whatever your Cloud Instance IP is
# Change Remote Port to whatever your service on your cloud instance is
# If Local Host does not work, use the service IP directly
REMOTE_MQTT_HOST = "44.203.53.203"
REMOTE_MQTT_PORT = 31583
REMOTE_MQTT_TOPIC = "mosquitto-face"

LOCAL_MQTT_HOST = "mosquitto-service"
LOCAL_MQTT_PORT = 1883
LOCAL_MQTT_TOPIC = "mosquitto-face"

def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))
        client.subscribe(LOCAL_MQTT_TOPIC)

def on_connect_remote(client, userdata, flags, rc):
        print("connected to remote with rc: " + str(rc))

def on_message(client, userdata, msg):
        try:
                print("message received: ")
                msg = msg.payload
                remote_mqttclient.publish(REMOTE_MQTT_TOPIC, payload=msg, qos=0, retain=False)
        except:
                print("Unexpected error:", sys.exc_info()[0])

print("Remote Client")
remote_mqttclient = mqtt.Client()
remote_mqttclient.on_connect = on_connect_remote
remote_mqttclient.connect(REMOTE_MQTT_HOST, REMOTE_MQTT_PORT, 60)
remote_mqttclient.loop_start()

print("Local Client")
local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)

local_mqttclient.on_message = on_message
local_mqttclient.loop_forever()
