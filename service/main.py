from fastapi import FastAPI
from service.api.api import main_router

app = FastAPI(project_name= "Emotion Detection")
app.include_router(main_router)

@app.get('/')
async def root():
    return {'hello':'hii'}