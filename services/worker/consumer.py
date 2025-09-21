# services/worker/consumer.py
import os, json
from pathlib import Path
from collections import deque, defaultdict

import numpy as np
import requests
import paho.mqtt.client as mqtt

# --------- Config / Env ---------
MQTT_HOST = os.getenv("MQTT_HOST", "mqtt-broker")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
# Tüm bataryaları dinle: battery/<BAT_ID>
TOPIC_IN  = os.getenv("TOPIC_IN", "battery/+")
API_URL   = os.getenv("API_URL", "http://api:8000/predict")

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
META_PATH = MODELS_DIR / "meta.json"

# meta.json ör.: {"window":30, "features":["voltage_v","current_a","temp_c"]}
meta = json.loads(META_PATH.read_text(encoding="utf-8"))
W = int(meta.get("window", 30))
DEFAULT_FEATURES = meta.get("features", ["voltage_v", "current_a", "temp_c"])

# Mühendislik özellikleri (opsiyonel)
USE_ENGINEERED = os.getenv("USE_ENGINEERED", "0") == "1"
MA_N = int(os.getenv("MA_N", "5"))  # rolling pencere

# Her batarya için ayrı pencere buffer’ı
buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=W))


# --------- Feature engineering helpers ---------
def _rolling_ma_std(x: np.ndarray, n: int):
    """Basit rolling mean/std; uçlarda kısa pencere kullanır."""
    ma = np.zeros_like(x, dtype=np.float32)
    sd = np.zeros_like(x, dtype=np.float32)
    for i in range(len(x)):
        s = x[max(0, i - n + 1): i + 1]
        ma[i] = s.mean()
        sd[i] = s.std() if len(s) > 1 else 0.0
    return ma, sd


def make_window_matrix(buf: deque) -> tuple[np.ndarray, list[str]]:
    """
    buf -> (W,F) matris ve feature isimleri.
    Giriş buf elemanı: [V, I, T]
    """
    arr = np.asarray(buf, dtype=np.float32)  # (W,3) [V,I,T]
    if not USE_ENGINEERED:
        return arr, ["voltage_v", "current_a", "temp_c"]

    V, I, T = arr[:, 0], arr[:, 1], arr[:, 2]

    # dV/dt ~ ardışık fark (publisher sabit dt ile veri gönderiyor)
    dV = np.zeros_like(V, dtype=np.float32)
    if len(V) > 1:
        dV[1:] = np.diff(V)

    V_ma, V_std = _rolling_ma_std(V, MA_N)
    I_ma, I_std = _rolling_ma_std(I, MA_N)
    T_ma, T_std = _rolling_ma_std(T, MA_N)

    feats = [
        "voltage_v", "current_a", "temp_c",
        "dV", "V_ma", "V_std", "I_ma", "I_std", "T_ma", "T_std",
    ]
    X = np.column_stack([V, I, T, dV, V_ma, V_std, I_ma, I_std, T_ma, T_std]).astype(np.float32)
    return X, feats


# --------- MQTT callbacks ---------
def on_connect(client: mqtt.Client, userdata, flags, rc, properties=None):
    print("[worker] connected rc=", rc, flush=True)
    client.subscribe(TOPIC_IN, qos=1)


def on_message(client: mqtt.Client, userdata, msg: mqtt.MQTTMessage):
    """
    Beklenen payload (publisher):
      {"t": <float>, "voltage": <float>, "current": <float>, "temp": <float>}
    Giriş topic:  battery/<BAT_ID>
    Çıkış topic:  battery/<BAT_ID>/pred
    """
    try:
        data = json.loads(msg.payload.decode("utf-8"))
        V = data.get("voltage")
        I = data.get("current")
        T = data.get("temp")
        if V is None or I is None or T is None:
            return

        # batarya id'yi güvenli şekilde çıkar
        parts = msg.topic.split("/")
        bat_id = parts[1] if len(parts) >= 2 and parts[0] == "battery" else "UNKNOWN"

        # pencereye ekle
        buf = buffers[bat_id]
        buf.append([float(V), float(I), float(T)])

        # pencere dolduysa API'ye gönder
        if len(buf) == W:
            X, feats = make_window_matrix(buf)
            payload = {"window": X.tolist(), "features": feats, "horizon": 1}

            try:
                r = requests.post(API_URL, json=payload, timeout=5)
                r.raise_for_status()
                pred = r.json()
            except Exception as e:
                print("[worker] API error:", e, flush=True)
                return

            out_topic = f"battery/{bat_id}/pred"
            client.publish(out_topic, json.dumps(pred), qos=1)

            # Not: deque(maxlen=W) sayesinde pencere doğal olarak kayan yapıda.
            # Ekstra pop gerekmiyor.

        # İsteğe bağlı debug (ilk mesajlar):
        # if len(buf) < 5:
        #     print(f"[worker] {bat_id} buf_len={len(buf)} V={V:.3f} I={I:.3f} T={T:.2f}", flush=True)

    except Exception as e:
        print("[worker] error:", e, flush=True)


# --------- Run ---------
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_HOST, MQTT_PORT, 60)
client.loop_forever()
