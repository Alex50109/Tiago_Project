import zlib
import cv2
import numpy as np
import torch
from flask import Flask, request, Response
from depth_anything_v2.dpt import DepthAnythingV2

app = Flask(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vitl' # or 'vits', 'vitb'
dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20 # 20 for indoor model, 80 for outdoor model

model = DepthAnythingV2(**{**model_configs[encoder], "max_depth": max_depth})
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
model.to(DEVICE).eval()

@app.route('/predict_depth_raw', methods=['POST'])
def predict_depth_raw():
    file_bytes = np.frombuffer(request.data, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return Response("Invalid image data", status=400)

    depth = model.infer_image(image).astype(np.float32)
    h, w = depth.shape

    print(f"Server-side Depth Stats | Min: {depth.min():.3f}, Max: {depth.max():.3f}, Mean: {depth.mean():.3f}")

    depth_contiguous = np.ascontiguousarray(depth, dtype=np.float32)

    raw_bytes = depth_contiguous.tobytes()
    compressed_bytes = zlib.compress(raw_bytes, level=1)

    return Response(
        compressed_bytes,
        mimetype='application/octet-stream',
        headers={
            'X-Depth-Height': str(h),
            'X-Depth-Width': str(w)
        }
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
