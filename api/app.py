from fastapi import FastAPI
from api.lifespan import lifespan
from api import batch_router
from api import single_router




   
app= FastAPI(lifespan=lifespan)

app.include_router(batch_router.router)
app.include_router(single_router.router)







@app.get("/health")
def health_check():
    return {"status": "healthy"}
    
    

    