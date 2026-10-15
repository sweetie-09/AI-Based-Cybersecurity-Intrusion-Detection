import math
import statistics
import pandas as pd


def safe_mean(values):
    return statistics.mean(values) if values else 0.0


def safe_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def safe_max(values):
    return max(values) if values else 0.0


def safe_min(values):
    return min(values) if values else 0.0


def safe_sum(values):
    return sum(values) if values else 0.0


def extract_features(flow):

    duration = (
        flow["last_time"] - flow["start_time"]
    )

    duration_us = duration * 1_000_000

    fwd_lengths = flow["fwd_lengths"]
    bwd_lengths = flow["bwd_lengths"]

    all_lengths = fwd_lengths + bwd_lengths

    fwd_iats = flow["fwd_iats"]
    bwd_iats = flow["bwd_iats"]

    total_packets = (
        len(fwd_lengths) +
        len(bwd_lengths)
    )

    total_bytes = (
        sum(fwd_lengths) +
        sum(bwd_lengths)
    )

    if duration > 0:

        flow_bytes_sec = total_bytes / duration
        flow_packets_sec = total_packets / duration

        fwd_packets_sec = len(fwd_lengths) / duration
        bwd_packets_sec = len(bwd_lengths) / duration

    else:

        flow_bytes_sec = 0
        flow_packets_sec = 0
        fwd_packets_sec = 0
        bwd_packets_sec = 0

    row = {

        "Destination Port":
            flow["first_dst_port"],

        "Flow Duration":
            duration_us,

        "Total Fwd Packets":
            len(fwd_lengths),

        "Total Backward Packets":
            len(bwd_lengths),

        "Total Length of Fwd Packets":
            sum(fwd_lengths),

        "Total Length of Bwd Packets":
            sum(bwd_lengths),

        "Fwd Packet Length Max":
            safe_max(fwd_lengths),

        "Fwd Packet Length Min":
            safe_min(fwd_lengths),

        "Fwd Packet Length Mean":
            safe_mean(fwd_lengths),

        "Fwd Packet Length Std":
            safe_std(fwd_lengths),

        "Bwd Packet Length Max":
            safe_max(bwd_lengths),

        "Bwd Packet Length Min":
            safe_min(bwd_lengths),

        "Bwd Packet Length Mean":
            safe_mean(bwd_lengths),

        "Bwd Packet Length Std":
            safe_std(bwd_lengths),

        "Flow Bytes/s":
            flow_bytes_sec,

        "Flow Packets/s":
            flow_packets_sec,

        "Flow IAT Mean":
            safe_mean(fwd_iats + bwd_iats) * 1_000_000,

        "Flow IAT Std":
            safe_std(fwd_iats + bwd_iats) * 1_000_000,

        "Flow IAT Max":
            safe_max(fwd_iats + bwd_iats) * 1_000_000,

        "Flow IAT Min":
            safe_min(fwd_iats + bwd_iats) * 1_000_000,

        "Fwd IAT Total":
            safe_sum(fwd_iats) * 1_000_000,

        "Fwd IAT Mean":
            safe_mean(fwd_iats) * 1_000_000,

        "Fwd IAT Std":
            safe_std(fwd_iats) * 1_000_000,

        "Fwd IAT Max":
            safe_max(fwd_iats) * 1_000_000,

        "Fwd IAT Min":
            safe_min(fwd_iats) * 1_000_000,

        "Bwd IAT Total":
            safe_sum(bwd_iats) * 1_000_000,

        "Bwd IAT Mean":
            safe_mean(bwd_iats) * 1_000_000,

        "Bwd IAT Std":
            safe_std(bwd_iats) * 1_000_000,

        "Bwd IAT Max":
            safe_max(bwd_iats) * 1_000_000,

        "Bwd IAT Min":
            safe_min(bwd_iats) * 1_000_000,

        "Fwd PSH Flags":
            0,

        "Bwd PSH Flags":
            0,

        "Fwd URG Flags":
            flow["urg"],

        "Bwd URG Flags":
            0,

        "Fwd Header Length":
            safe_sum(flow["fwd_header_lengths"]),

        "Bwd Header Length":
            safe_sum(flow["bwd_header_lengths"]),

        "Fwd Packets/s":
            fwd_packets_sec,

        "Bwd Packets/s":
            bwd_packets_sec,

        "Min Packet Length":
            safe_min(all_lengths),

        "Max Packet Length":
            safe_max(all_lengths),

        "Packet Length Mean":
            safe_mean(all_lengths),

        "Packet Length Std":
            safe_std(all_lengths),

        "Packet Length Variance":
            statistics.variance(all_lengths)
            if len(all_lengths) > 1 else 0.0,

        "FIN Flag Count":
            flow["fin"],

        "SYN Flag Count":
            flow["syn"],

        "RST Flag Count":
            flow["rst"],

        "PSH Flag Count":
            flow["psh"],

        "ACK Flag Count":
            flow["ack"],

        "URG Flag Count":
            flow["urg"],

        "CWE Flag Count":
            0,

        "ECE Flag Count":
            0,

        "Down/Up Ratio":
            len(bwd_lengths) / len(fwd_lengths)
            if len(fwd_lengths) > 0 else 0,

        "Average Packet Size":
            safe_mean(all_lengths),

        "Avg Fwd Segment Size":
            safe_mean(fwd_lengths),

        "Avg Bwd Segment Size":
            safe_mean(bwd_lengths),

        "Fwd Header Length.1":
            safe_sum(flow["fwd_header_lengths"]),

        "Fwd Avg Bytes/Bulk":
            0,

        "Fwd Avg Packets/Bulk":
            0,

        "Fwd Avg Bulk Rate":
            0,

        "Bwd Avg Bytes/Bulk":
            0,

        "Bwd Avg Packets/Bulk":
            0,

        "Bwd Avg Bulk Rate":
            0,

        "Subflow Fwd Packets":
            len(fwd_lengths),

        "Subflow Fwd Bytes":
            sum(fwd_lengths),

        "Subflow Bwd Packets":
            len(bwd_lengths),

        "Subflow Bwd Bytes":
            sum(bwd_lengths),

        "Init_Win_bytes_forward":
            flow["fwd_windows"][0]
            if flow["fwd_windows"] else 0,

        "Init_Win_bytes_backward":
            flow["bwd_windows"][0]
            if flow["bwd_windows"] else 0,

        "act_data_pkt_fwd":
            sum(
                1 for x in flow["fwd_payload_lengths"]
                if x > 0
            ),

        "min_seg_size_forward":
            safe_min(flow["fwd_payload_lengths"]),

        "Active Mean":
            0,

        "Active Std":
            0,

        "Active Max":
            0,

        "Active Min":
            0,

        "Idle Mean":
            0,

        "Idle Std":
            0,

        "Idle Max":
            0,

        "Idle Min":
            0
    }

    # Replace invalid numerical values
    for key, value in row.items():

        if isinstance(value, float):

            if math.isnan(value) or math.isinf(value):
                row[key] = 0.0

    return row


def flow_to_dataframe(flow):

    features = extract_features(flow)

    return pd.DataFrame([features])