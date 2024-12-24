import asyncio
import os
import pathlib
import json

import torch
import cv2
import numpy as np

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame

##########################################
# YOLO MODEL (örnek)
##########################################
pathlib.PosixPath = pathlib.WindowsPath

print("Loading YOLO model...")
model = torch.hub.load(
    'yolov5_model',
    'custom',
    path='yolov5_model/weights/best.pt',
    source='local',
    force_reload=True
)
print("Model loaded.")

# Global bir MediaRelay, aynı track'i birden fazla ekleme durumunda kullanışlı.
media_relay = MediaRelay()


class YoloTransformTrack(MediaStreamTrack):
    """
    Tarayıcıdan gelen video track'i yakala.
    - YOLO tespiti
    - bounding box'ları görüntü üzerine çiz
    - işlenmiş kareyi geri döndür (böylece remote track'te gözükür)
    - tespit bilgisini dataChannel üzerinden tarayıcıya JSON olarak gönder
    """
    kind = "video"

    def _init_(self, track, data_channel):
        super()._init_()
        self.track = track
        self.data_channel = data_channel

    async def recv(self):
        # Orijinal kareyi al
        frame = await self.track.recv()

        # frame'i ndarray'e dönüştür (BGR24)
        img = frame.to_ndarray(format="bgr24")

        # YOLO inference
        results = model(img)
        labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        detections = []
        for label, cord in zip(labels, cords):
            x1, y1, x2, y2, conf = cord
            if conf > 0.3:
                x1 = int(x1 * img.shape[1])
                y1 = int(y1 * img.shape[0])
                x2 = int(x2 * img.shape[1])
                y2 = int(y2 * img.shape[0])
                label_name = results.names[int(label)]
                detections.append({
                    "label": label_name,
                    "bbox": [x1, y1, x2, y2],
                    "conf": float(conf.item())
                })

                # BBox çiz
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    label_name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (36, 255, 12),
                    2
                )

        # DataChannel'a JSON mesaj
        if self.data_channel and self.data_channel.readyState == "open":
            msg = {
                "type": "detections",
                "objects": detections
            }
            self.data_channel.send(json.dumps(msg))

        # İşlenmiş kareyi geri döndür (remote track'te bounding box'lı video göreceğiz)
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


async def index(request):
    """ index.html içeriklerini döndürür """
    here = os.path.dirname(_file_)
    with open(os.path.join(here, "index.html"), "r", encoding="utf-8") as f:
        html_content = f.read()
    return web.Response(content_type="text/html", text=html_content)


async def offer(request):
    """ Tarayıcıdan gelen Offer'ı al, Answer üret, YOLO track'i ekle """
    params = await request.json()

    pc = RTCPeerConnection()
    data_channel = None

    @pc.on("datachannel")
    def on_datachannel(channel):
        nonlocal data_channel
        data_channel = channel
        print("DataChannel received on server:", channel.label)

    @pc.on("track")
    def on_track(track):
        print("Track received:", track.kind)
        if track.kind == "video":
            # Relay edelim (aiortc best practice)
            relay_track = media_relay.subscribe(track)
            # YOLO transform track
            yolo_track = YoloTransformTrack(relay_track, data_channel)
            # Sunucudan geriye video track gönderiyoruz
            pc.addTrack(yolo_track)

    # Remote description
    offer_obj = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    await pc.setRemoteDescription(offer_obj)

    # Answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })


async def run_server(host="0.0.0.0", port=8080):
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    print(f"Starting server on {host}:{port}")
    await site.start()

    # Devamlı çalış
    while True:
        await asyncio.sleep(3600)


if _name_ == "_main_":
    asyncio.run(run_server())
