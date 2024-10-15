import io
import json

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

app = FastAPI()

model = YOLO("yolo_model/yolov10s.pt")

@app.post("/object_detect")
async def create_upload_files(file: UploadFile):
    try:
        is_person = False

        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")


        results = model(image)

        annotated_image = results[0].to_json()

        for annotated_object in json.loads(annotated_image):
            if annotated_object['name'] == 'person':
                is_person = True

        return JSONResponse(content={"result": is_person})
    except Exception as e:
        print(e)
        return JSONResponse(content={"result": 'fail'})
