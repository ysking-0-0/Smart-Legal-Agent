
import requests
import os
import time

url = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
fname = "C:\Users\86157\.cache\chroma\onnx_models\all-MiniLM-L6-v2\onnx.tar.gz"
current_size = 22724608

headers = {'Range': f'bytes={current_size}-'} if current_size > 0 else {}
mode = 'ab' if current_size > 0 else 'wb'

session = requests.Session()
session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))

try:
    response = session.get(url, headers=headers, stream=True, timeout=120)
    total_size = int(response.headers.get('content-length', 0)) + current_size
    
    with open(fname, mode) as f:
        downloaded = current_size
        for chunk in response.iter_content(chunk_size=16384):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
except Exception as e:
    pass
