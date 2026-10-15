from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict
import time
import statistics


flows = {}


def make_flow_key(packet):
    ip = packet[IP]

    src_ip = ip.src
    dst_ip = ip.dst

    if TCP in packet:
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
        protocol = 6

    elif UDP in packet:
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport
        protocol = 17

    else:
        src_port = 0
        dst_port = 0
        protocol = ip.proto

    endpoint1 = (src_ip, src_port)
    endpoint2 = (dst_ip, dst_port)

    if endpoint1 <= endpoint2:
        return (endpoint1, endpoint2, protocol)

    return (endpoint2, endpoint1, protocol)


def process_packet(packet):

    if IP not in packet:
        return

    if not (TCP in packet or UDP in packet):
        return

    ip = packet[IP]

    timestamp = float(packet.time)
    packet_length = len(packet)

    key = make_flow_key(packet)

    if key not in flows:

        if TCP in packet:
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
            protocol = 6
        else:
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
            protocol = 17

        flows[key] = {
            "start_time": timestamp,
            "last_time": timestamp,

            "first_src_ip": ip.src,
            "first_dst_ip": ip.dst,
            "first_src_port": src_port,
            "first_dst_port": dst_port,

            "protocol": protocol,

            "fwd_packets": [],
            "bwd_packets": [],

            "fwd_lengths": [],
            "bwd_lengths": [],

            "fwd_iats": [],
            "bwd_iats": [],

            "last_fwd_time": None,
            "last_bwd_time": None,

            "syn": 0,
            "ack": 0,
            "fin": 0,
            "rst": 0,
            "psh": 0,
            "urg": 0,

            "fwd_syn": 0,
            "bwd_syn": 0,

            "fwd_ack": 0,
            "bwd_ack": 0,

            "fwd_header_lengths": [],
            "bwd_header_lengths": [],

            "fwd_payload_lengths": [],
            "bwd_payload_lengths": [],

            "fwd_windows": [],
            "bwd_windows": [],

            "fwd_active_times": [],
            "bwd_active_times": []
        }

    flow = flows[key]

    flow["last_time"] = timestamp

    is_forward = (
        ip.src == flow["first_src_ip"]
        and ip.dst == flow["first_dst_ip"]
    )

    if is_forward:

        packet_list = flow["fwd_packets"]
        lengths = flow["fwd_lengths"]

        if flow["last_fwd_time"] is not None:
            flow["fwd_iats"].append(
                timestamp - flow["last_fwd_time"]
            )

        flow["last_fwd_time"] = timestamp

        packet_list.append(packet)
        lengths.append(packet_length)

        if IP in packet:
            flow["fwd_header_lengths"].append(
                packet[IP].ihl * 4
            )

        if TCP in packet:

            flow["fwd_payload_lengths"].append(
                len(packet[TCP].payload)
            )

            flow["fwd_windows"].append(
                packet[TCP].window
            )

            flags = packet[TCP].flags

            if "S" in flags:
                flow["fwd_syn"] += 1

            if "A" in flags:
                flow["fwd_ack"] += 1

        elif UDP in packet:

            flow["fwd_payload_lengths"].append(
                len(packet[UDP].payload)
            )

    else:

        packet_list = flow["bwd_packets"]
        lengths = flow["bwd_lengths"]

        if flow["last_bwd_time"] is not None:
            flow["bwd_iats"].append(
                timestamp - flow["last_bwd_time"]
            )

        flow["last_bwd_time"] = timestamp

        packet_list.append(packet)
        lengths.append(packet_length)

        if IP in packet:
            flow["bwd_header_lengths"].append(
                packet[IP].ihl * 4
            )

        if TCP in packet:

            flow["bwd_payload_lengths"].append(
                len(packet[TCP].payload)
            )

            flow["bwd_windows"].append(
                packet[TCP].window
            )

            flags = packet[TCP].flags

            if "S" in flags:
                flow["bwd_syn"] += 1

            if "A" in flags:
                flow["bwd_ack"] += 1

        elif UDP in packet:

            flow["bwd_payload_lengths"].append(
                len(packet[UDP].payload)
            )

    if TCP in packet:

        flags = packet[TCP].flags

        if "S" in flags:
            flow["syn"] += 1

        if "A" in flags:
            flow["ack"] += 1

        if "F" in flags:
            flow["fin"] += 1

        if "R" in flags:
            flow["rst"] += 1

        if "P" in flags:
            flow["psh"] += 1

        if "U" in flags:
            flow["urg"] += 1


def show_current_flows():

    print("\n" + "=" * 70)
    print("CURRENT LIVE FLOWS")
    print("=" * 70)

    print("Total flows:", len(flows))

    for i, (key, flow) in enumerate(flows.items(), 1):

        duration = (
            flow["last_time"] - flow["start_time"]
        )

        print(
            f"\nFlow {i}"
            f"\nProtocol       : {flow['protocol']}"
            f"\nDuration       : {duration:.6f} sec"
            f"\nFWD packets    : {len(flow['fwd_packets'])}"
            f"\nBWD packets    : {len(flow['bwd_packets'])}"
            f"\nFWD bytes      : {sum(flow['fwd_lengths'])}"
            f"\nBWD bytes      : {sum(flow['bwd_lengths'])}"
        )


print("=" * 70)
print("LIVE BIDIRECTIONAL FLOW BUILDER")
print("=" * 70)
print("Capturing 30 packets...")
print("Open your browser to generate traffic.\n")


sniff(
    prn=process_packet,
    count=30
)

show_current_flows()

print("\n" + "=" * 70)
print("FLOW COLLECTION COMPLETE")
print("=" * 70)