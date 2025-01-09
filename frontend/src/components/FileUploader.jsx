import React, { useState } from "react";
import axios from "axios";

function FileUploader({ setDehazedImageUrl, setHazyImageUrl }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setHazyImageUrl(URL.createObjectURL(selectedFile)); // Send hazy image preview to App
    }
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file to upload.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      const response = await axios.post(
        "http://127.0.0.1:8000/dehaze",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      setDehazedImageUrl(response.data.dehazed_image_url);
      setLoading(false);
    } catch (err) {
      setError("Error processing the image. Please try again.");
      setLoading(false);
    }
  };

  return (
    <div className="card shadow-sm p-4 mb-4">
      <h2 className="text-center mb-4">Upload Your Hazy Image</h2>
      <div className="mb-3">
        <label htmlFor="fileInput" className="form-label">
          Select Image
        </label>
        <input
          type="file"
          className="form-control"
          id="fileInput"
          onChange={handleFileChange}
        />
      </div>
      <div className="d-grid gap-2">
        <button
          className="btn btn-primary"
          onClick={handleUpload}
          disabled={loading}
        >
          {loading ? (
            <span>
              <span
                className="spinner-border spinner-border-sm me-2"
                role="status"
                aria-hidden="true"
              ></span>
              Processing...
            </span>
          ) : (
            "Upload and Dehaze"
          )}
        </button>
      </div>
      {error && <div className="alert alert-danger mt-3">{error}</div>}
    </div>
  );
}

export default FileUploader;
