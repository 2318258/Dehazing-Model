import React, { useState } from "react";
import FileUploader from "./components/FileUploader";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";

function App() {
  const [dehazedImageUrl, setDehazedImageUrl] = useState(null);
  const [hazyImageUrl, setHazyImageUrl] = useState(null); // To store hazy image preview

  return (
    <div>
      {/* Navbar Header */}
      <nav className="navbar navbar-expand-lg navbar-dark bg-primary">
        <div className="container">
          <a className="navbar-brand m-auto" href="/">
            Image Dehazing App
          </a>
        </div>
      </nav>

      {/* Main Content */}
      <div className="container my-5">
        <FileUploader
          setDehazedImageUrl={setDehazedImageUrl}
          setHazyImageUrl={setHazyImageUrl}
        />
        {dehazedImageUrl && hazyImageUrl && (
          <div className="row mt-5">
            {/* Hazy Image */}
            <div className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header text-center">
                  <h5>Hazy Image</h5>
                </div>
                <div className="card-body d-flex justify-content-center align-items-center">
                  <img
                    src={hazyImageUrl}
                    alt="Hazy Preview"
                    className="img-fluid fixed-image-size"
                  />
                </div>
              </div>
            </div>
            {/* Dehazed Image */}
            <div className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header text-center">
                  <h5>Dehazed Image</h5>
                </div>
                <div className="card-body d-flex justify-content-center align-items-center">
                  <img
                    src={`http://127.0.0.1:8000${dehazedImageUrl}`}
                    alt="Dehazed Output"
                    className="img-fluid fixed-image-size"
                  />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
