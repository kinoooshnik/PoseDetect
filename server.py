import cgi
#!/usr/env python3
import http.server
import io
import json
import socketserver
import threading
import urllib.parse
import uuid

from PIL import Image

from posedetect import check_pose_from_pil

PORT = 1336


shared_dict = {
    'test': {
        'status': 'ok',
        'data': {
            'dots': [[[0, 1]]],
            'check': [False]
        }
    }
}


def do_job(pil, id):
    shared_dict[id] = {
        'status': 'ok',
        'message': 'processing'
    }
    try:
        (dots, check) = check_pose_from_pil(pil)
        actual_dots = [list(x.values()) for x in dots]
        shared_dict[id] = {
            'status': 'ok',
            'data': {
                'dots': actual_dots,
                'check': check
            }
        }
    except Exception as e:
        shared_dict[id] = {
            'status': 'fail',
            'message': 'fail during pose check'
        }


class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    timeout = 1

    def send_headers_and_body(self, body):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode('utf-8'))

    def do_GET(self):
        data = {'status': 'fail', 'message': 'id not found'}
        try:
            id = urllib.parse.parse_qs(self.path)['/result?id'][0]
            data = shared_dict[id]
        except Exception as e:
            pass

        body = json.dumps(data)
        self.send_headers_and_body(body)

    def do_POST(self):
        generated_id = str(uuid.uuid4())
        res, info = self.deal_post_data(generated_id)

        print(res, info, "by: ", self.client_address)

        fail = json.dumps({'status': 'fail'})
        ok = json.dumps({
            'status': 'ok',
            'data': {
                'id': generated_id
            }
        })

        body = ok if res else fail
        self.send_headers_and_body(body)

    def deal_post_data(self, id):
        try:
            ctype, pdict = cgi.parse_header(self.headers['Content-Type'])
            pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
            pdict['CONTENT-LENGTH'] = int(self.headers['Content-Length'])

            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={
                'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': self.headers['Content-Type'], })

            buf = form["file"].file.read()
            pil = Image.open(io.BytesIO(buf))

            thread = threading.Thread(
                target=do_job, args=(pil, id))
            thread.daemon = True
            thread.start()

        except Exception as e:
            return (False, "Error during uploading")
        return (True, "Files uploaded")


Handler = CustomHTTPRequestHandler
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.socket.close()
