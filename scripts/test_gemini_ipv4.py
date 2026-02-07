"""Test Gemini API connectivity with forced IPv4."""
import socket
import ssl
import urllib.request
import sys

# Force IPv4 for socket connections
original_getaddrinfo = socket.getaddrinfo

def getaddrinfo_ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    return original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

socket.getaddrinfo = getaddrinfo_ipv4_only

print("Testing Gemini API (IPv4 forced)...")
try:
    url = "https://generativelanguage.googleapis.com"
    context = ssl.create_default_context()
    req = urllib.request.Request(url, headers={'User-Agent': 'Python-Test'})
    with urllib.request.urlopen(req, timeout=10, context=context) as response:
        print(f"✅ SUCCESS! Status: {response.getcode()}")
        print("Connectivity is working.")
except urllib.error.HTTPError as e:
    # 404 is actually a success here (server reached)
    print(f"✅ SUCCESS! Reached server (Status {e.code})")
    print("Connectivity is working.")
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)
