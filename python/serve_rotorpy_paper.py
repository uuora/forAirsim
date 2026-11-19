from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial
from pathlib import Path
import json
from argparse import ArgumentParser

class Viewer(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, '.mjs': 'text/javascript'}

    def do_GET(self):
        if self.path != '/progress.json':
            return super().do_GET()
        rows = [json.loads(p.read_text(encoding='utf-8')) for p in Path(self.directory).glob('route_*/*/*/result.json')]
        data = json.dumps(rows).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

if __name__ == '__main__':
    parser = ArgumentParser(description='Serve the local RotorPy replay page')
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    root = args.campaign.resolve(strict=True)
    with ThreadingHTTPServer(('127.0.0.1', args.port), partial(Viewer, directory=str(root))) as server:
        print(f'http://127.0.0.1:{args.port}/', flush=True)
        server.serve_forever()
