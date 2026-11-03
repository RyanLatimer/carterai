#!/usr/bin/env python3
"""CI-only CONNECT proxy + local PAC server. No request bodies or credentials are logged."""
import http.server, socket, socketserver, select, sys
log = sys.argv[1]
class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        line = self.rfile.readline().decode('ascii').strip()
        if not line: return
        method, target, _ = line.split(' ',2)
        while self.rfile.readline() not in (b'\r\n',b'\n',b''): pass
        if method != 'CONNECT':
            self.wfile.write(b'HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n'); return
        host, port = target.rsplit(':',1)
        with open(log,'a') as out: out.write('CONNECT '+host+'\n')
        try: upstream=socket.create_connection((host,int(port)),timeout=20)
        except OSError:
            self.wfile.write(b'HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n'); return
        with upstream:
            self.wfile.write(b'HTTP/1.1 200 Connection Established\r\n\r\n'); self.wfile.flush()
            for stream in (upstream,self.connection): stream.settimeout(60)
            while True:
                ready,_,_=select.select([upstream,self.connection],[],[],30)
                if not ready: return
                for source in ready:
                    data=source.recv(65536)
                    if not data: return
                    (upstream if source is self.connection else self.connection).sendall(data)
class Proxy(socketserver.ThreadingTCPServer):
    allow_reuse_address=True
    daemon_threads=True
class PAC(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body=b'function FindProxyForURL(url, host) { return "PROXY 127.0.0.1:18081"; }'
        self.send_response(200); self.send_header('Content-Type','application/x-ns-proxy-autoconfig')
        self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self,*args): pass
import threading
threading.Thread(target=lambda: http.server.ThreadingHTTPServer(('127.0.0.1',18082),PAC).serve_forever(),daemon=True).start()
Proxy(('127.0.0.1',18081),Handler).serve_forever()
