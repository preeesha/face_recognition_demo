import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import LiveRecognition from "./components/LiveRecognition";
import Recording from "./components/Recording";
import "./App.css";

function App() {
  const webcamRef = useRef(null);
  const [personName, setPersonName] = useState("");
  const [capturedImages, setCapturedImages] = useState([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const [activeTab, setActiveTab] = useState("capture");

  const captureImage = () => {
    if (!personName) {
      alert("Please enter a name first!");
      return;
    }
    const imageSrc = webcamRef.current.getScreenshot();
    setCapturedImages((prev) => [...prev, imageSrc]);
  };

  const startAutoCapture = async (count = 20, interval = 500) => {
    if (!personName) {
      alert("Please enter a name first!");
      return;
    }

    setIsCapturing(true);
    setCapturedImages([]);
    setIsDone(false);

    for (let i = 0; i < count; i++) {
      const imageSrc = webcamRef.current.getScreenshot();
      setCapturedImages((prev) => [...prev, imageSrc]);

      try {
        await axios.post("http://localhost:8080/upload", {
          name: personName,
          image: imageSrc,
          index: i + 1,
        });
      } catch (err) {
        console.error("Upload failed:", err);
      }

      await new Promise((res) => setTimeout(res, interval));
    }

    setIsCapturing(false);
    alert(`✅ Done capturing 20 images for ${personName}`);

    try {
      await axios.post("http://localhost:8080/send-to-pi", { name: personName });
      alert(`📦 Dataset for ${personName} sent to Raspberry Pi successfully!`);
    } catch (err) {
      alert("❌ Failed to send dataset to Raspberry Pi: " + err.message);
    }

    setIsDone(true);
  };

  const resetForNewPerson = () => {
    setPersonName("");
    setCapturedImages([]);
    setIsDone(false);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <div className="header">
        <h1>🤖 Face Recognition System</h1>
        <p>Raspberry Pi + React Web Interface</p>
      </div>

      {/* Tab Navigation */}
      <div className="tabs">
        <button
          onClick={() => setActiveTab("capture")}
          className={`tab-btn ${activeTab === "capture" ? "active" : ""}`}
        >
          📸 Capture Dataset
        </button>
        <button
          onClick={() => setActiveTab("recognition")}
          className={`tab-btn ${activeTab === "recognition" ? "active" : ""}`}
        >
          🎥 Live Recognition
        </button>
        <button
          onClick={() => setActiveTab("recording")}
          className={`tab-btn ${activeTab === "recording" ? "active" : ""}`}
        >
          🎬 Recording
        </button>
      </div>

      {/* Capture Tab */}
      {activeTab === "capture" && (
        <div className="capture-container">
          <h2>📸 Dataset Capture (Web Version)</h2>

          {!isDone && (
            <input
              type="text"
              placeholder="Enter Person Name"
              value={personName}
              onChange={(e) => setPersonName(e.target.value)}
              className="name-input"
            />
          )}

          <div className="webcam-wrapper">
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              width={480}
              height={360}
              videoConstraints={{ facingMode: "user" }}
              className="webcam-feed"
            />
          </div>

          {!isDone ? (
            <div className="btn-group">
              <button
                onClick={captureImage}
                disabled={!personName || isCapturing}
                className={`btn ${!personName || isCapturing ? "disabled" : "capture-btn"}`}
              >
                Capture Image
              </button>
              <button
                onClick={() => startAutoCapture(20, 500)}
                disabled={isCapturing || !personName}
                className={`btn ${isCapturing || !personName ? "disabled" : "auto-btn"}`}
              >
                {isCapturing ? "Capturing..." : "Auto Capture (20 images)"}
              </button>
            </div>
          ) : (
            <div className="done-container">
              <h3 className="done-text">
                ✅ Done capturing 20 images for{" "}
                <span className="person-name">{personName}</span>!
              </h3>
              <button onClick={resetForNewPerson} className="btn secondary-btn">
                📸 Capture Other Person
              </button>
              <button
                onClick={async () => {
                  try {
                    await fetch("http://192.168.1.213:8000/build_database", {
                      method: "POST",
                    });
                    alert("✅ Database build started on Raspberry Pi!");
                  } catch (err) {
                    alert("❌ Failed to start database build: " + err.message);
                  }
                }}
                className="btn success-btn"
              >
                🗄️ Add Person to Database
              </button>
            </div>
          )}

          <h3 className="capture-count">
            Captured Images ({capturedImages.length})
          </h3>
          <div className="image-grid">
            {capturedImages.map((img, i) => (
              <img key={i} src={img} alt={`capture-${i}`} className="captured-img" />
            ))}
          </div>
        </div>
      )}

      {activeTab === "recognition" && <LiveRecognition />}
      {activeTab === "recording" && <Recording />}
    </div>
  );
}

export default App;
