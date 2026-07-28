import socket
import sys
import time
import threading

def server(ip, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((ip, port))
        s.listen(1)
        print(f"[SERVER] Listening on {ip}:{port}", flush=True)
        conn, addr = s.accept()
        print(f"[SERVER] Accepted connection from {addr}", flush=True)
        data = conn.recv(1024)
        print(f"[SERVER] Received data: {data.decode()}", flush=True)
        conn.sendall(b"Hello client")
        conn.close()
        s.close()
    except Exception as e:
        print(f"[SERVER ERROR] {e}", flush=True)

def client(ip, port):
    try:
        time.sleep(2)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"[CLIENT] Connecting to {ip}:{port}", flush=True)
        s.connect((ip, port))
        s.sendall(b"Hello server")
        data = s.recv(1024)
        print(f"[CLIENT] Received data: {data.decode()}", flush=True)
        s.close()
    except Exception as e:
        print(f"[CLIENT ERROR] {e}", flush=True)

if __name__ == "__main__":
    ip = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = 45678
    
    t1 = threading.Thread(target=server, args=(ip, port))
    t2 = threading.Thread(target=client, args=(ip, port))
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    print("Done!", flush=True)
