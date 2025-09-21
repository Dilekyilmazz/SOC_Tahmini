# services/ui/app.py
import os, json, time, threading
from collections import deque
from datetime import datetime

import pandas as pd
import streamlit as st
import paho.mqtt.client as mqtt

# --------- Config ---------
MQTT_HOST = os.getenv("MQTT_HOST", "mqtt-broker")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC_SUB = os.getenv("TOPIC_SUB", "battery/+/pred")  # tüm pillerin tahmin konuları

# --------- Session State (kalıcı) ---------
if "hist_by_battery" not in st.session_state:
    # pil_id -> deque[(ts, soc)]
    st.session_state.hist_by_battery = {}
if "last_ts_by_battery" not in st.session_state:
    # pil_id -> son mesaj ISO time
    st.session_state.last_ts_by_battery = {}
if "lock" not in st.session_state:
    st.session_state.lock = threading.Lock()

hist_by_battery = st.session_state.hist_by_battery
last_ts_by_battery = st.session_state.last_ts_by_battery
lock = st.session_state.lock


# --------- MQTT Callbacks ---------
def on_connect(c, u, flags, rc, *extra):
    print("[ui] mqtt connected rc=", rc, flush=True)
    # Hem verilen TOPIC_SUB'a hem de emniyet için wildcard'a abone ol
    c.subscribe(TOPIC_SUB)
    c.subscribe("battery/+/pred")

from datetime import datetime

def on_message(c, u, msg):
    try:
        data = json.loads(msg.payload.decode("utf-8"))
        soc = float(data.get("soc_pred", 0.0))
        ts = datetime.now().astimezone()   # <- string değil datetime

        parts = msg.topic.split("/")
        bat = parts[1] if len(parts) >= 3 else "UNKNOWN"

        with lock:
            dq = hist_by_battery.get(bat)
            if dq is None:
                from collections import deque
                dq = deque(maxlen=4000)
                hist_by_battery[bat] = dq
            dq.append((ts, soc))
            last_ts_by_battery[bat] = ts   # <- datetime olarak tut
    except Exception as e:
        print("[ui] parse error:", e, flush=True)



# --------- MQTT Client (kalıcı) ---------
if "client" not in st.session_state:
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    c.enable_logger()
    c.on_connect = on_connect
    c.on_message = on_message
    c.connect(MQTT_HOST, MQTT_PORT, 60)
    c.loop_start()
    # callback gelmeden de abone ol (çifte emniyet)
    c.subscribe(TOPIC_SUB)
    c.subscribe("battery/+/pred")
    st.session_state.client = c
client = st.session_state.client


# --------- UI ---------
st.set_page_config(page_title="SoC Live Dashboard", layout="wide")
st.title("🔋 SoC Live Dashboard")

topic_note = f"{TOPIC_SUB} (listening all batteries)" if "+" in TOPIC_SUB else TOPIC_SUB
st.caption(f"MQTT: {MQTT_HOST}:{MQTT_PORT} • Topic: {topic_note}")

with lock:
    bat_list = sorted(hist_by_battery.keys())

sel = st.selectbox(
    "Battery",
    bat_list,
    index=0 if bat_list else None,
    placeholder="(waiting data)"
)

col1, col2 = st.columns([3, 1])

# Grafik (yalnızca seçilen pil)
with col1:
    with lock:
        if sel and hist_by_battery.get(sel):
            df = pd.DataFrame(hist_by_battery[sel], columns=["ts", "soc"])
            df["ts"] = pd.to_datetime(df["ts"])
            df = df.set_index("ts")
        else:
            df = pd.DataFrame({"soc": []})
    st.line_chart(df)

# Metrikler (seçilen pil için)
# Metrikler (seçilen pil için)
with col2:
    with lock:
        dq = hist_by_battery.get(sel, [])
        last_dt = dq[-1][0] if dq else None      # datetime
        last    = dq[-1][1] if dq else None
        received_sel = len(dq)
        connected = client.is_connected()

    st.metric("Current SoC", f"{last*100:.1f}%" if last is not None else "–")
    st.caption(f"Connected: {connected} • Received (selected): {received_sel}")

    if last_dt:
        st.caption(f"Last update: {last_dt.strftime('%Y-%m-%d %H:%M:%S')}")



# (isteğe bağlı) tüm pillerin özet tablosu
with lock:
    if hist_by_battery:
        rows = [{
            "battery": b,
            "received": len(dq),
            "last": (last_ts_by_battery.get(b).strftime('%Y-%m-%d %H:%M:%S')
                     if last_ts_by_battery.get(b) else "–")
        } for b, dq in hist_by_battery.items()]
        st.dataframe(pd.DataFrame(rows).set_index("battery"),
                     use_container_width=True, height=160)

# Yumuşak auto-refresh
time.sleep(1.0)
st.rerun()
