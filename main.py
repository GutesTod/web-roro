from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from ultralytics import YOLO
import numpy as np
import io
from PIL import Image, ImageDraw
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Загрузите вашу модель YOLO
model = YOLO("yolo11n.pt")  # Замените на путь к вашей модели

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
