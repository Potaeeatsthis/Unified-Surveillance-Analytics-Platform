import nest_asyncio
import os
import cv2
import json
import time
from ultralytics import YOLO
from ultralytics.solutions import TrackZone
import subprocess
import numpy as np
import io
import aiohttp
import asyncio
from ultralytics.utils import LOGGER
from urllib.parse import urlparse
import logging
LOGGER.setLevel(logging.CRITICAL)
from setproctitle import setproctitle
setproctitle("people-pipeline-trt-int8")

INPUT_RTSP_SERVER=os.getenv("INPUT_RTSP_SERVER") 
OUTPUT_RTSP_SERVER=os.getenv("OUTPUT_RTSP_SERVER") 
UPLOAD_URL=os.getenv("UPLOAD_URL") 
SITE_NAME=os.getenv("SITE_NAME") 
CAMERA_NAME=os.getenv("CAMERA_NAME") 
POLYGONS=os.getenv("POLYGONS")
CROP_SIZE=os.getenv("CROP_SIZE")
MODEL=os.getenv("MODEL") 
CONF=os.getenv("CONF")
CLASSES=os.getenv("CLASSES")
AUTH_TOKEN=os.getenv("AUTH_TOKEN")F
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720 

nest_asyncio.apply()

def start_ffmpeg_input(rtsp_input_url):
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
    return subprocess.Popen(rtsp_stream_command, stdin=subprocess.PIPE)

async def upload_file_async(image_data, site_name, camera_name, url, token, track_id):
    payload = aiohttp.FormData()
    payload.add_field('site_name', site_name)
    payload.add_field('camera_name', camera_name)
    payload.add_field(
        'file',
        image_data.getvalue(),
        filename='cropped_image.jpg',
        content_type='image/jpg'
    )

    headers = {'Authorization': f'Bearer {token}'}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=payload) as response:
                if response.status == 201:
                    res = await response.json()
                    res["track_id"] = track_id
                    print(res)
                    return res
                else:
                    print(f"Failed to upload: {response.status}")
                    return None
    except Exception as e:
        print(f"Error in async upload_file: {e}")

def check_intersection(bbox, trackzone_points):
    try:
        x1, y1, x2, y2 = bbox
        bbox_polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        trackzone_polygon = np.array(trackzone_points)
        intersection = cv2.intersectConvexConvex(bbox_polygon.astype(np.float32), trackzone_polygon.astype(np.float32))
        return intersection[0] > 0
    except Exception as e:
        print(f"Error in check_intersection: {e}")
        return False

def convert_polygon_to_points(polygons):
    region_points = []
    for polygon in polygons:
        points = [(int(point["x"]), int(point["y"])) for point in polygon["content"]]
        region_points.append(points)
    return region_points

region_points = convert_polygon_to_points(json.loads(POLYGONS))

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
    silent=False,
    verbose=False
)

uploaded_track_ids = set()
track_age = {}
track_frames = {}

ffmpeg_process = start_ffmpeg_input(INPUT_RTSP_SERVER)
process1 = start_ffmpeg_output(OUTPUT_RTSP_SERVER)

async def main():
    global ffmpeg_process, process1, track_age, track_frames, uploaded_track_ids

    current_time = time.time()
    if frame_count % 100 == 0:
    tracks_to_delete = []
    for track_id, last_seen_time in track_last_seen.items():
        if current_time - last_seen_time > 300:
            tracks_to_delete.append(track_id)

    for track_id in tracks_to_delete:
        del track_age[track_id]
        del track_frames[track_id]
        del track_last_seen[track_id]
        uploaded_track_ids.discard(track_id) 
    
    while True:
        raw_frame = ffmpeg_process.stdout.read(FRAME_WIDTH * FRAME_HEIGHT * 3)
        if len(raw_frame) != FRAME_WIDTH * FRAME_HEIGHT * 3:
            print("Frame read incomplete. Restarting input process...")
            ffmpeg_process.terminate()
            ffmpeg_process = start_ffmpeg_input(INPUT_RTSP_SERVER)
            continue

        frame = np.frombuffer(raw_frame, np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
        frame = frame.copy()
        resizeFrame = cv2.resize(frame, (640, 480))

        try:
            tracked_results = trackzone.trackzone(frame)
        except Exception as e:
            print(f"Tracking error: {e}")
            continue

        for bbox, track_id in zip(trackzone.boxes, trackzone.track_ids):
            if track_id not in track_age:
                track_age[track_id] = 0
                track_frames[track_id] = None  

            track_age[track_id] += 1
            track_frames[track_id] = resizeFrame

            print("track_id", track_id, ":", track_age[track_id])

            if track_age[track_id] >= int(FRAME_INDEX_CROP) and track_id not in uploaded_track_ids:
                uploaded_track_ids.add(track_id)
                
                results = trackzone.model.predict(resizeFrame)

                detected_objects = results[0].boxes.data.tolist()

                for detection in detected_objects:
                    x1, y1, x2, y2, conf, class_id = detection[:6]

                    x1, y1, x2, y2 = convert_coordinates(int(x1), int(y1), int(x2), int(y2))
                    
                    bbox = (x1, y1, x2, y2)

                    if check_intersection(bbox, region_points):
                        track_id = detection[6] if len(detection) > 6 else None
                        if track_id is None:
                            continue

                        if track_id not in track_age:
                            track_age[track_id] = 0
                            track_frames[track_id] = None  

                        track_age[track_id] += 1

                        width = x2 - x1
                        height = y2 - y1
                        max_side = max(width, height)

                        expand = int(max_side * 0.2)
                        max_side += expand
                        
                        x_center = (x1 + x2) // 2
                        y_center = (y1 + y2) // 2

                        x1 = int(max(0, x_center - max_side // 2))
                        y1 = int(max(0, y_center - max_side // 2))
                        x2 = int(min(frame.shape[1], x_center + max_side // 2))
                        y2 = int(min(frame.shape[0], y_center + max_side // 2))

                        cropped_img = frame[y1:y2, x1:x2]
                    
                        _, buffer = cv2.imencode('.jpg', cropped_img)
                        image_data = io.BytesIO(buffer.tobytes())

                        asyncio.create_task(
                            upload_file_async(
                                image_data=image_data,
                                site_name=SITE_NAME,
                                camera_name=CAMERA_NAME,
                                url=UPLOAD_URL,
                                token=AUTH_TOKEN,
                                track_id=track_id
                            )
                        )
                        del track_age[track_id]
                        del track_frames[track_id]

        try:
            process1.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print("Output stream broken. Restarting output process...")
            process1.terminate()
            process1 = start_ffmpeg_output(OUTPUT_RTSP_SERVER)

        await asyncio.sleep(0.01)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        ffmpeg_process.stdout.close()
        ffmpeg_process.terminate()
        process1.stdin.close()
        process1.terminate()
        cv2.destroyAllWindows()
