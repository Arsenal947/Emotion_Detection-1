from fastapi import APIRouter, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
import numpy as np
from service.core.schemas.output import APIOutput
from service.core.logic.onnx_inference import emotions_detecter

detect_router = APIRouter()
@detect_router.post("/detect", response_model=APIOutput)

async def detect(im: UploadFile):
    if im.filename.split(".")[-1] in ("jpg","png","jpeg"):
        pass
    else:
        raise HTTPException(status_code=415, detail="This file type is not supported.")
    image = Image.open(BytesIO(im.file.read()))
    image = np.array(image)

    return emotions_detecter(image)