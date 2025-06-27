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
from ultralytics.solutions import TrackZone
from ultralytics.utils import LOGGER
import logging
from urllib.parse import urlparse

LOGGER.setLevel(logging.CRITICAL)
from setproctitle import setproctitle
setproctitle("vehicle-pipeline-trt-int8")

INPUT_RTSP_SERVER = os.getenv("INPUT_RTSP_SERVER")
OUTPUT_RTSP_SERVER = os.getenv("OUTPUT_RTSP_SERVER")
UPLOAD_URL = os.getenv("UPLOAD_URL")
SITE_NAME = os.getenv("SITE_NAME")
CAMERA_NAME = os.getenv("CAMERA_NAME")
LINE_A = os.getenv("LINE_A")
LINE_B = os.getenv("LINE_B")
POLYGONS = os.getenv("POLYGONS")
MODEL = os.getenv("MODEL")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
CONF = os.getenv("CONF", "0.45")
CLASSES = os.getenv("CLASSES", '[2]')
FRAME_INDEX_CROP = os.getenv("FRAME_INDEX_CROP", "10")
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

def validate_config():
    """Validates loaded configuration to ensure the script can run safely."""
    required_vars = {
        "INPUT_RTSP_SERVER": INPUT_RTSP_SERVER,
        "OUTPUT_RTSP_SERVER": OUTPUT_RTSP_SERVER,
        "UPLOAD_URL": UPLOAD_URL,
        "SITE_NAME": SITE_NAME,
        "CAMERA_NAME": CAMERA_NAME,
        "LINE_A": LINE_A,
        "LINE_B": LINE_B,
        "POLYGONS": POLYGONS,
        "MODEL": MODEL,
        "AUTH_TOKEN": AUTH_TOKEN,
    }
    for var_name, value in required_vars.items():
        if not value:
            print(f"FATAL: Environment variable '{var_name}' is not set.", file=sys.stderr)
            sys.exit(1)
    try:
        urlparse(INPUT_RTSP_SERVER)
        urlparse(OUTPUT_RTSP_SERVER)
        urlparse(UPLOAD_URL)
        json.loads(POLYGONS)
        json.loads(LINE_A)
        json.loads(LINE_B)
        json.loads(CLASSES)
        float(CONF)
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

async def upload_file_async(image_data, site_name, camera_name, url, token, track_id):
    payload = aiohttp.FormData()
    payload.add_field('site_name', site_name)
    payload.add_field('camera_name', camera_name)
    payload.add_field(
        'file',
        image_data.getvalue(),
        filename='cropped_image.jpg',
        content_type='image/jpeg'
    )
    headers = {'Authorization': f'Bearer {token}'}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=payload) as response:
                if response.status == 201:
                    res = await response.json()
                    res["track_id"] = track_id
                    print(f"Upload success for track_id {track_id}: {res}")
                    return res
                else:
                    error_text = await response.text()
                    print(f"Failed to upload for track_id {track_id}: {response.status} - {error_text}", file=sys.stderr)
                    return None
    except Exception as e:
        print(f"Error in async upload_file for track_id {track_id}: {e}", file=sys.stderr)

def convert_polygon_to_points(polygons):
    polygons_data = json.loads(polygons)
    region_points = []
    for polygon in polygons_data:
        points = [(int(point["x"]), int(point["y"])) for point in polygon["content"]]
        region_points.append(points)
    return region_points

region_points = convert_polygon_to_points(POLYGONS)
line_a_points = [(int(pt["x"]), int(pt["y"])) for pt in json.loads(LINE_A)]
line_b_points = [(int(pt["x"]), int(pt["y"])) for pt in json.loads(LINE_B)]
line_a_thresh = (line_a_points[0][1] + line_a_points[1][1]) / 2
line_b_thresh = (line_b_points[0][1] + line_b_points[1][1]) / 2

def convert_coordinates(x1, y1, x2, y2):
    scale_x = FRAME_WIDTH / 640
    scale_y = FRAME_HEIGHT / 480
    return int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

trackzone = TrackZone(
    show=False,
    region=region_points,
    model=MODEL,
    classes=json.loads(CLASSES),
    conf=float(CONF),
    silent=True,
    verbose=False
)

uploaded_track_ids = set()
track_age = {}
track_line_state = {}
track_last_seen = {}
TRACK_TIMEOUT_SECONDS = 300

ffmpeg_process = start_ffmpeg_input(INPUT_RTSP_SERVER)
process1 = start_ffmpeg_output(OUTPUT_RTSP_SERVER)

async def main():
    global ffmpeg_process, process1, track_line_state, track_age, uploaded_track_ids, track_last_seen
    frame_counter = 0
    while True:
        raw_frame = ffmpeg_process.stdout.read(FRAME_WIDTH * FRAME_HEIGHT * 3)
        if len(raw_frame) != FRAME_WIDTH * FRAME_HEIGHT * 3:
            print("Frame read incomplete. Restarting input process...", file=sys.stderr)
            ffmpeg_process.terminate()
            time.sleep(5)
            ffmpeg_process = start_ffmpeg_input(INPUT_RTSP_SERVER)
            continue

        frame = np.frombuffer(raw_frame, np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3)).copy()
        resizeFrame = cv2.resize(frame, (640, 480))

        try:
            trackzone.trackzone(resizeFrame)
        except Exception as e:
            print(f"Tracking error: {e}", file=sys.stderr)
            continue
        
        current_time = time.time()
        active_track_ids = set()

        for bbox, track_id in zip(trackzone.boxes, trackzone.track_ids):
            active_track_ids.add(track_id)
            track_last_seen[track_id] = current_time

            xmin, ymin, xmax, ymax = convert_coordinates(*map(int, bbox))
            center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2

            if track_id not in track_age:
                track_age[track_id] = 0
            track_age[track_id] += 1

            if track_age[track_id] < int(FRAME_INDEX_CROP):
                continue
            
            state = track_line_state.setdefault(track_id, {
                "line_a_crossed": False,
                "line_b_crossed": False,
                "prev_center_y": center_y,
            })
            prev_center_y = state["prev_center_y"]

            if not state["line_a_crossed"] and prev_center_y < line_a_thresh and center_y >= line_a_thresh:
                 state["line_a_crossed"] = True
                 print(f"Track {track_id} crossed Line A.")
            
            if state["line_a_crossed"] and not state["line_b_crossed"] and prev_center_y < line_b_thresh and center_y >= line_b_thresh:
                state["line_b_crossed"] = True

                if track_id not in uploaded_track_ids:
                    print(f"Track {track_id} crossed Line B. Scheduling upload.")
                    uploaded_track_ids.add(track_id)
                    
                    half_size = 150
                    x1, y1 = max(0, center_x - half_size), max(0, center_y - half_size)
                    x2, y2 = min(frame.shape[1], x1 + (half_size * 2)), min(frame.shape[0], y1 + (half_size*2))
                    cropped_image = frame[y1:y2, x1:x2]

                    _, buffer = cv2.imencode('.jpg', cropped_image)
                    image_data = io.BytesIO(buffer.tobytes())
                    
                    asyncio.create_task(
                        upload_file_async(
                            image_data=image_data, site_name=SITE_NAME, camera_name=CAMERA_NAME,
                            url=UPLOAD_URL, token=AUTH_TOKEN, track_id=track_id
                        )
                    )

            state["prev_center_y"] = center_y

        frame_counter += 1
        if frame_counter % 900 == 0:
            tracks_to_delete = [
                track_id for track_id, last_seen in track_last_seen.items()
                if current_time - last_seen > TRACK_TIMEOUT_SECONDS
            ]
            if tracks_to_delete:
                print(f"Cleaning up {len(tracks_to_delete)} stale track(s)...")
                for track_id in tracks_to_delete:
                    track_age.pop(track_id, None)
                    track_line_state.pop(track_id, None)
                    track_last_seen.pop(track_id, None)
                    uploaded_track_ids.discard(track_id)

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
