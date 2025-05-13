import paho.mqtt.publish as publish
auth = {'username': 'esp32user', 'password': 'red123'}

publish.single("display/text", "Hello secure world!",
               hostname="192.168.1.231", port=1883, auth=auth)

