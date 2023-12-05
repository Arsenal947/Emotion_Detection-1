from fastapi import APIRouter
from service.api.enpoints.detect import detect_router
from service.api.enpoints.test import test_router

main_router = APIRouter()

main_router.include_router(detect_router)
main_router.include_router(test_router)