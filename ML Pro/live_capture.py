from scapy.all import sniff

print("=" * 50)
print("REAL-TIME NETWORK PACKET CAPTURE")
print("=" * 50)
print("Waiting for network packets...")
print("Use your browser for a few seconds.")
print("Press Ctrl+C to stop.\n")


def show_packet(packet):
    print(packet.summary())


sniff(
    prn=show_packet,
    count=20
)

print("\nCapture finished.")