import nest_asyncio
import os
import sys
import cv2
import json
import time
import subprocess
import numpy as np
import io
import aiohttp
import asyncio
from ultralytics import YOLO
from ultralytics.solutions import Heatmap
from ultralytics.utils import LOGGER
import logging
from urllib.parse import urlparse

LOGGER.setLevel(logging.CRITICAL)
from setproctitle import setproctitle
setproctitle("crowd-pipeline-trt-int8")

INPUT_RTSP_SERVER = os.getenv("INPUT_RTSP_SERVER")
OUTPUT_RTSP_SERVER = os.getenv("OUTPUT_RTSP_SERVER")
UPLOAD_URL = os.getenv("UPLOAD_URL")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
SITE_NAME = os.getenv("SITE_NAME")
CAMERA_NAME = os.getenv("CAMERA_NAME")
MODEL = os.getenv("MODEL")
CONF = float(os.getenv("CONF", "0.75"))
CLASSES = os.getenv("CLASSES", '[0, 1, 2, 3, 5, 7]')
ACCUMULATION_TIME = int(os.getenv("ACCUMULATION_TIME", "60"))
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

def validate_config():
    required_vars = {
        "INPUT_RTSP_SERVER": INPUT_RTSP_SERVER,
        "OUTPUT_RTSP_SERVER": OUTPUT_RTSP_SERVER,
        "UPLOAD_URL": UPLOAD_URL,
        "AUTH_TOKEN": AUTH_TOKEN,
        "SITE_NAME": SITE_NAME,
        "CAMERA_NAME": CAMERA_NAME,
        "MODEL": MODEL,
    }
    for var_name, value in required_vars.items():
        if not value:
            print(f"FATAL: Environment variable '{var_name}' is not set.", file=sys.stderr)
            sys.exit(1)
    try:
        urlparse(INPUT_RTSP_SERVER)
        urlparse(OUTPUT_RTSP_SERVER)
        urlparse(UPLOAD_URL)
        json.loads(CLASSES)
    except Exception as e:
        print(f"FATAL: Error parsing environment variables. Check formats. Details: {e}", file=sys.stderr)
        sys.exit(1)

validate_config()
nest_asyncio.apply()

def start_ffmpeg_input(rtsp_input_url):
    print(f"Starting FFmpeg input stream from: {rtsp_input_url}")
    return subprocess.Popen(
        [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-hwaccel", "cuda",
            "-i", rtsp_input_url,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vf", f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
            "-r", "30",
            "-an", "-"
        ],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=2**20
    )

def start_ffmpeg_output(output_rtsp_server):
    print(f"Starting FFmpeg output stream to: {output_rtsp_server}")
    rtsp_stream_command = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pixel_format", "bgr24",
        "-video_size", f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
        "-r", "30",
        "-i", "-",
        "-c:v", "libx264",
        "-rtsp_transport", "tcp",
        "-f", "rtsp",
        "-tune", "zerolatency",
        "-preset", "veryfast",
        output_rtsp_server
    ]
    return subprocess.Popen(rtsp_stream_command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

async def predict_model_and_get_bbox(frame, yolo_model):
    try:
        results = yolo_model.predict(frame, verbose=False)
        counts = {"person": 0, "bicycle": 0, "car": 0, "motorbike": 0, "bus": 0, "truck": 0}
        bboxes = []
        class_to_key = {0: "person", 1: "bicycle", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

        if len(results[0].boxes) > 0:
            for result in results[0].boxes:
                class_id = int(result.cls[0].item())
                if class_id in class_to_key:
                    counts[class_to_key[class_id]] += 1
                    bbox = result.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)
                    confidence = float(result.conf[0].item())
                    bbox_info = {
                        "class": class_to_key[class_id],
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence
                    }
                    bboxes.append(bbox_info)
        return counts, bboxes
    except Exception as e:
        print(f"Error in predict_model_and_get_bbox: {e}", file=sys.stderr)
        return {"person": 0, "bicycle": 0, "car": 0, "motorbike": 0, "bus": 0, "truck": 0}, []

async def upload_file_async(image_data, site_name, camera_name, url, token, counts, bboxes):
    payload = aiohttp.FormData()
    payload.add_field('site_name', site_name)
    payload.add_field('camera_name', camera_name)
    payload.add_field('person', str(counts.get('person', 0)))
    payload.add_field('bicycle', str(counts.get('bicycle', 0)))
    payload.add_field('car', str(counts.get('car', 0)))
    payload.add_field('motorbike', str(counts.get('motorbike', 0)))
    payload.add_field('bus', str(counts.get('bus', 0)))
    payload.add_field('truck', str(counts.get('truck', 0)))
    payload.add_field('bbbox', json.dumps(bboxes))
    payload.add_field(
        'file',
        image_data.getvalue(),
        filename='heatmap.jpg',
        content_type='image/jpeg'
    )
    headers = {'Authorization': f'Bearer {token}'}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=payload) as response:
                if response.status == 201:
                    res = await response.json()
                    print(f"Upload successful. Response: {res}")
                    return res
                else:
                    error_text = await response.text()
                    print(f"Failed to upload heatmap: {response.status} - {error_text}", file=sys.stderr)
                    return None
    except Exception as e:
        print(f"Error in async upload_file: {e}", file=sys.stderr)

heatmap = Heatmap(
    show=False,
    model=MODEL,
    classes=json.loads(CLASSES),
    conf=float(CONF),
    colormap=cv2.COLORMAP_JET,
    silent=True,
    verbose=False
)

ffmpeg_process = start_ffmpeg_input(INPUT_RTSP_SERVER)
process1 = start_ffmpeg_output(OUTPUT_RTSP_SERVER)

async def main():
    global ffmpeg_process, process1
    yolo_model = YOLO(MODEL)
    frame_accumulator = []
    start_time = time.time()

    while True:
        raw_frame = ffmpeg_process.stdout.read(FRAME_WIDTH * FRAME_HEIGHT * 3)
        if not raw_frame or len(raw_frame) != FRAME_WIDTH * FRAME_HEIGHT * 3:
            print("Frame read error or stream ended. Reconnecting...", file=sys.stderr)
            ffmpeg_process.terminate()
            time.sleep(5)
            ffmpeg_process = start_ffmpeg_input(INPUT_RTSP_SERVER)
            continue

        frame = np.frombuffer(raw_frame, np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3)).copy()
        
        max_frames = ACCUMULATION_TIME * 30 + 60 
        if len(frame_accumulator) < max_frames:
             frame_accumulator.append(frame)
        else:
             frame_accumulator.pop(0)
             frame_accumulator.append(frame)

        elapsed_time = time.time() - start_time

        if elapsed_time >= ACCUMULATION_TIME:
            print(f"Accumulation time reached. Processing {len(frame_accumulator)} frames...")
            counts, bboxes = await predict_model_and_get_bbox(frame, yolo_model)
            print(f"Object counts: {counts}")

            cumulative_heatmap = None
            for acc_frame in frame_accumulator:
                current_heatmap = heatmap.generate_heatmap(acc_frame)
                if current_heatmap is not None:
                    if cumulative_heatmap is None:
                        cumulative_heatmap = current_heatmap.astype(np.float32)
                    else:
                        cumulative_heatmap = cv2.addWeighted(cumulative_heatmap, 0.95, current_heatmap.astype(np.float32), 0.05, 0)
            
            if cumulative_heatmap is not None:
                cumulative_heatmap = cv2.normalize(cumulative_heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                _, buffer = cv2.imencode('.jpg', cumulative_heatmap)
                image_data = io.BytesIO(buffer.tobytes())
                
                asyncio.create_task(
                    upload_file_async(
                        image_data=image_data, site_name=SITE_NAME, camera_name=CAMERA_NAME,
                        url=UPLOAD_URL, token=AUTH_TOKEN, counts=counts, bboxes=bboxes
                    )
                )
            
            frame_accumulator = []
            start_time = time.time()

        try:
            process1.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            print("Output stream broken. Restarting output process...", file=sys.stderr)
            process1.terminate()
            time.sleep(5)
            process1 = start_ffmpeg_output(OUTPUT_RTSP_SERVER)

        await asyncio.sleep(0.001)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if 'ffmpeg_process' in globals() and ffmpeg_process.poll() is None:
            ffmpeg_process.terminate()
        if 'process1' in globals() and process1.poll() is None:
            process1.terminate()
        cv2.destroyAllWindows()
        print("Shutdown complete.")
