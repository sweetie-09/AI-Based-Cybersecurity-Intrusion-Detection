import json
import os
import time
import joblib
import pandas as pd

from scapy.all import sniff
from flow_builder import process_packet, flows
from feature_extractor import extract_features


MODEL_PATH = "models/multiclass_model.pkl"
RESULT_FILE = "results/live_results.json"

os.makedirs("results", exist_ok=True)

model = joblib.load(MODEL_PATH)

processed_flows = set()

stats = {
    "packets": 0,
    "flows": 0,
    "normal": 0,
    "attacks": 0,
    "last_prediction": "Waiting...",
    "last_confidence": 0,
    "last_source": "-",
    "last_destination": "-",
    "last_protocol": "-",
    "last_time": "-",
    "logs": [],
    "started_at": time.time()
}


def save_results():

    with open(RESULT_FILE, "w") as f:
        json.dump(stats, f, indent=4)


def predict_flow(flow_key, flow):

    if flow_key in processed_flows:
        return

    total_packets = (
        len(flow["fwd_packets"]) +
        len(flow["bwd_packets"])
    )

    if total_packets < 2:
        return

    try:

        features = extract_features(flow)

        df = pd.DataFrame([features])

        expected_columns = model.feature_names_in_

        for column in expected_columns:

            if column not in df.columns:
                df[column] = 0

        df = df[expected_columns]

        prediction = model.predict(df)[0]

        # Confidence
        confidence = 0.0

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(df)[0]
            confidence = float(max(probabilities)) * 100

        source = (
            f"{flow['first_src_ip']}:{flow['first_src_port']}"
        )

        destination = (
            f"{flow['first_dst_ip']}:{flow['first_dst_port']}"
        )

        protocol_number = flow["protocol"]

        if protocol_number == 6:
            protocol = "TCP"
        elif protocol_number == 17:
            protocol = "UDP"
        else:
            protocol = str(protocol_number)

        current_time = time.strftime("%H:%M:%S")

        if str(prediction).upper() == "BENIGN":

            status = "NORMAL"
            stats["normal"] += 1

        else:

            status = "ATTACK"
            stats["attacks"] += 1

        stats["flows"] = len(flows)

        stats["last_prediction"] = str(prediction)
        stats["last_confidence"] = round(confidence, 2)
        stats["last_source"] = source
        stats["last_destination"] = destination
        stats["last_protocol"] = protocol
        stats["last_time"] = current_time

        log = {
            "time": current_time,
            "source": source,
            "destination": destination,
            "protocol": protocol,
            "prediction": str(prediction),
            "confidence": round(confidence, 2),
            "status": status
        }

        stats["logs"].insert(0, log)

        # Keep only latest 50
        stats["logs"] = stats["logs"][:50]

        processed_flows.add(flow_key)

        save_results()

        print(
            f"[{current_time}] "
            f"{source} -> {destination} | "
            f"{protocol} | "
            f"{prediction} | "
            f"{confidence:.2f}%"
        )

    except Exception as e:

        print("Prediction error:", e)


def packet_callback(packet):

    stats["packets"] += 1

    process_packet(packet)


print("=" * 70)
print("AI CYBERSECURITY LIVE MONITOR")
print("=" * 70)

save_results()

try:

    while True:

        sniff(
            prn=packet_callback,
            count=20,
            store=False
        )

        for key, flow in list(flows.items()):

            predict_flow(key, flow)

        stats["flows"] = len(flows)

        save_results()

except KeyboardInterrupt:

    print("\nLive monitoring stopped.")