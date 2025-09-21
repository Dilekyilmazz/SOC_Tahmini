# tools/mqtt_publisher.py
import argparse, time, json
import pandas as pd
import paho.mqtt.client as mqtt

ap = argparse.ArgumentParser()
ap.add_argument("--file", default="data/processed/B0006.parquet")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=1883)
ap.add_argument("--topic", default="battery/B0006")
ap.add_argument("--delay", type=float, default=0.2, help="mesajlar arası saniye")
args = ap.parse_args()

# Paho 2.x: versiyon belirle ve ağ döngüsünü başlat
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(args.host, args.port, 60)
client.loop_start()

df = pd.read_parquet(args.file)
# sadece discharge ve gerekli kolonlar
mask = (df["cycle_type"] == "discharge") & df[["voltage_v","current_a","temp_c"]].notna().all(axis=1)
df = df[mask][["t_s","voltage_v","current_a","temp_c"]]

for _, row in df.iterrows():
    msg = {
        "t": float(row["t_s"]),
        "voltage": float(row["voltage_v"]),
        "current": float(row["current_a"]),
        "temp": float(row["temp_c"]),
    }
    # qos=0 yeterli
    client.publish(args.topic, json.dumps(msg), qos=0)
    time.sleep(args.delay)

client.loop_stop()
client.disconnect()
