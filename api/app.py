from fastapi import FastAPI,status, Response
from api.lifespan import lifespan
from api import batch_router
from api import single_router
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST




   
app= FastAPI(lifespan=lifespan)

app.include_router(batch_router.router)
app.include_router(single_router.router)



@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy"}
    

@app.get("/metrics")
def metrics():
    return Response(generate_latest(),media_type=CONTENT_TYPE_LATEST)