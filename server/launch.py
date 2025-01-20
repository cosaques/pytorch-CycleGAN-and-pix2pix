import uvicorn
from options.test_options import TestOptions
from server.utils import merge_images, predict_professional_image, preprocess_image, load_model
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from ultralyticsplus import YOLO

# Define the lifespan handler
def lifespan(app: FastAPI):
    # Startup logic
    opt = TestOptions().parse()
    model_pic2pic = load_model(opt)
    model_yolo = YOLO('kesimeg/yolov8n-clothing-detection')
    app.state.model_pic2pic = model_pic2pic
    app.state.model_yolo = model_yolo

    yield  # Application lifespan runs here

    # Shutdown logic (optional)
    # Cleanup code goes here if needed (e.g., closing connections)

# Initialize the FastAPI app with the lifespan handler
app = FastAPI(lifespan=lifespan)

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/process_tshirt/")
async def process_tshirt(image: UploadFile = File(...)):
    # Check if the uploaded file is an image
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read the uploaded image
        img_bytes = await image.read()

        # Call the preprocessing function
        preprocessed_img = preprocess_image(
            app.state.model_yolo,
            img_bytes,
            "./datasets/t-shirts/1736765461_scaled.jpeg"
        )
        if not preprocessed_img:
            raise HTTPException(status_code=500, detail="No t-shirt found on photo")

        # Call the prediction function
        predicted_img = predict_professional_image(
            app.state.model_pic2pic,
            preprocessed_img,
            "./results/fake.jpg"
        )

        result_img = merge_images(preprocessed_img, predicted_img)

        # Convert the preprocessed image to a byte stream
        result_io = io.BytesIO()
        result_img.save(result_io, format="JPEG")
        result_io.seek(0)

        return StreamingResponse(result_io, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"health_check": "OK"}

if __name__ == '__main__':
    uvicorn.run("server.launch:app", host="127.0.0.1", port=8000, reload=True)
