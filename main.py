from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from ultralytics import YOLO
import numpy as np
import io
from PIL import Image, ImageDraw
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()

# Load models
model_yolo11n = YOLO("yolo11n.pt")
model_yolo11m = YOLO("yolo11m.pt")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class UploadData(BaseModel):
    model: str
    file: UploadFile

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_image(data: UploadData = Form(...)):
    file = data.file
    model_name = data.model

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Неверный тип файла")

    # Выберите соответствующую модель
    if model_name == "yolo11n":
        model = model_yolo11n
    elif model_name == "yolo11m":
        model = model_yolo11m
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
