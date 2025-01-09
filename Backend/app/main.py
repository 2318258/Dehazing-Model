from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.model import load_model, dehaze_image
import os
import shutil
from uuid import uuid4

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Directory to store uploaded and processed files
UPLOAD_DIR = "uploads"
DEHAZED_DIR = "dehazed"

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DEHAZED_DIR, exist_ok=True)

# Load the model at startup
model = load_model()

@app.post("/dehaze")
async def dehaze(file: UploadFile = File(...)):
    """
    API endpoint to upload an image and perform dehazing.

    Args:
        file (UploadFile): Uploaded image file.

    Returns:
        dict: JSON response containing the URL of the dehazed image.
    """
    try:
        # Save the uploaded file temporarily
        unique_id = uuid4().hex
        input_file_path = os.path.join(UPLOAD_DIR, f"{unique_id}_{file.filename}")
        with open(input_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Perform dehazing
        dehazed_file_path = os.path.join(DEHAZED_DIR, f"dehazed_{file.filename}")
        dehaze_image(input_file_path, model, output_path=dehazed_file_path)

        # Return the URL of the dehazed image
        return {
            "message": "Dehazing successful",
            "dehazed_image_url": f"/dehazed/{os.path.basename(dehazed_file_path)}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the uploaded file
        if os.path.exists(input_file_path):
            os.remove(input_file_path)

@app.get("/dehazed/{file_name}")
async def get_dehazed_image(file_name: str):
    """
    Endpoint to retrieve the dehazed image by file name.

    Args:
        file_name (str): Name of the dehazed image file.

    Returns:
        FileResponse: The dehazed image file.
    """
    file_path = os.path.join(DEHAZED_DIR, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
