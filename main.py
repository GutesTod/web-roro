from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, FileResponse
from ultralytics import YOLO
import numpy as np
import io
import os
from PIL import Image, ImageDraw
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import cv2
from fastapi.middleware.cors import CORSMiddleware
import base64
import asyncio
import time

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить все источники для тестирования
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
model_yolo11n = YOLO("yolo11n.pt")
model_yolov5n = YOLO("yolov5n.pt")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class UploadData(BaseModel):
    model: str
    file: UploadFile

class VideoBase(BaseModel):
    ...

class UploadResponseVideo(VideoBase):
    filename: str

class DowloadVideo(VideoBase):
    filename: str

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post('/upload/video', response_model=UploadResponseVideo)
async def upload_video(file: UploadFile = File(...), model: str = Form(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Файл является не видео-файлом!")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Выберите соответствующую модель
    if model == "yolo11n":
        yolo_model = model_yolo11n
    elif model == "yolov5n":
        yolo_model = model_yolov5n
    else:
        raise HTTPException(status_code=400, detail="Неверный выбор модели")

    # Обработка видео в фоновом режиме
    background_tasks.add_task(process_video, file_path, yolo_model)

    return JSONResponse(
        content={
            "filename": file.filename
        },
        status_code=200
    )

async def process_video(file_path, model):
    cap = cv2.VideoCapture(file_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = os.path.join(PROCESSED_DIR, f"processed_{os.path.basename(file_path)}")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, save=False, imgsz=1280, conf=0.34)
        has_objects = False
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                has_objects = True
        if has_objects:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            await manager.broadcast(frame_base64)

        current_time = time.time()
        if current_time - start_time >= 30:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            await manager.broadcast(frame_base64)
            start_time = current_time

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

def delete_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Файл {file_path} был удален.")

@app.post('/download/video', response_class=FileResponse)
async def download_video(request: DowloadVideo, background_tasks: BackgroundTasks):
    file_path = os.path.join(PROCESSED_DIR, request.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Файл не найден!")

    response = FileResponse(file_path, media_type="application/octet-stream", filename=request.filename)

    background_tasks.add_task(delete_file, file_path)

    return response

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/image")
async def upload_image(data: UploadData = Form(...)):
    file = data.file
    model_name = data.model

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Неверный тип файла")

    # Выберите соответствующую модель
    if model_name == "yolo11n":
        model = model_yolo11n
    elif model_name == "yolov5n":
        model = model_yolov5n
    else:
        raise HTTPException(status_code=400, detail="Неверный выбор модели")

    # Чтение изображения
    image = Image.open(file.file)
    image_np = np.array(image)

    # Выполнение предсказания
    results = model.predict(source=image_np, save=False, imgsz=1280, conf=0.34)

    # Получение bounding box и рисование их на изображении
    person_count = 0
    draw = ImageDraw.Draw(image)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            person_count += 1

    # Преобразование изображения в формат, подходящий для отправки
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    return StreamingResponse(image_bytes, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
