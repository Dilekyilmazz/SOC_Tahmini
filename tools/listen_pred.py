import argparse, json
import paho.mqtt.client as mqtt
ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=1883)
ap.add_argument("--topic", default="battery/B0006/pred")
args = ap.parse_args()
def on_msg(c,u,m):
    try: print("PRED:", json.loads(m.payload.decode()))
    except: print("PRED(raw):", m.topic, m.payload)
c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
c.on_message = on_msg
c.connect(args.host, args.port, 60)
c.subscribe(args.topic)
c.loop_forever()
