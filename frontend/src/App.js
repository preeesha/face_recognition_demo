import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import LiveRecognition from "./LiveRecognition"; 
import Recording from "./Recording";

function App() {
  const webcamRef = useRef(null);
  const [personName, setPersonName] = useState("");
  const [capturedImages, setCapturedImages] = useState([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const [activeTab, setActiveTab] = useState("capture"); // "capture" | "recognition" | "recording"

  // Capture a single image
  const captureImage = () => {
    if (!personName) {
      alert("Please enter a name first!");
      return;
    }
    const imageSrc = webcamRef.current.getScreenshot();
    setCapturedImages((prev) => [...prev, imageSrc]);
  };

  // Auto-capture multiple frames (20 by default)
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

      // Send to backend
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

    // Send dataset to Raspberry Pi
    try {
      await axios.post("http://localhost:8080/send-to-pi", { name: personName });
      alert(`📦 Dataset for ${personName} sent to Raspberry Pi successfully!`);
    } catch (err) {
      alert("❌ Failed to send dataset to Raspberry Pi: " + err.message);
    }
    setIsDone(true);
  };

  // Reset for a new person
  const resetForNewPerson = () => {
    setPersonName("");
    setCapturedImages([]);
    setIsDone(false);
  };

  return (
    <div style={{ minHeight: "100vh", background: "#f5f5f5" }}>
      {/* Header */}
      <div
        style={{
          background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
          padding: "20px",
          textAlign: "center",
          color: "white",
          boxShadow: "0 2px 10px rgba(0,0,0,0.1)",
        }}
      >
        <h1 style={{ margin: 0, fontSize: "32px" }}>
          🤖 Face Recognition System
        </h1>
        <p style={{ margin: "5px 0 0 0", opacity: 0.9 }}>
          Raspberry Pi + React Web Interface
        </p>
      </div>

      {/* Tab Navigation */}
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          gap: "10px",
          padding: "20px",
          background: "#fff",
          borderBottom: "2px solid #e0e0e0",
        }}
      >
        <button
          onClick={() => setActiveTab("capture")}
          style={{
            padding: "12px 30px",
            fontSize: "16px",
            fontWeight: "bold",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            background: activeTab === "capture" ? "#667eea" : "#e0e0e0",
            color: activeTab === "capture" ? "white" : "#666",
            transition: "all 0.3s",
          }}
        >
          📸 Capture Dataset
        </button>
        <button
          onClick={() => setActiveTab("recognition")}
          style={{
            padding: "12px 30px",
            fontSize: "16px",
            fontWeight: "bold",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            background: activeTab === "recognition" ? "#667eea" : "#e0e0e0",
            color: activeTab === "recognition" ? "white" : "#666",
            transition: "all 0.3s",
          }}
        >
          🎥 Live Recognition
        </button>
        <button
          onClick={() => setActiveTab("recording")}
          style={{
            padding: "12px 30px",
            fontSize: "16px",
            fontWeight: "bold",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            background: activeTab === "recording" ? "#667eea" : "#e0e0e0",
            color: activeTab === "recording" ? "white" : "#666",
            transition: "all 0.3s",
          }}
        >
          🎬 Recording
        </button>
      </div>

      {/* Tab Content */}
      {activeTab === "capture" && (
        <div style={{ textAlign: "center", padding: "20px" }}>
          <h2>📸 Dataset Capture (Web Version)</h2>

          {!isDone && (
            <input
              type="text"
              placeholder="Enter Person Name"
              value={personName}
              onChange={(e) => setPersonName(e.target.value)}
              style={{
                padding: "12px",
                margin: "10px",
                width: "250px",
                fontSize: "16px",
                borderRadius: "8px",
                border: "2px solid #ddd",
              }}
            />
          )}

          <div style={{ display: "flex", justifyContent: "center", marginTop: "20px" }}>
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              width={480}
              height={360}
              videoConstraints={{ facingMode: "user" }}
              style={{ borderRadius: "8px", border: "3px solid #667eea" }}
            />
          </div>

          {!isDone ? (
            <div style={{ marginTop: "20px" }}>
              <button
                onClick={captureImage}
                disabled={!personName || isCapturing}
                style={{
                  padding: "12px 24px",
                  fontSize: "16px",
                  marginRight: "10px",
                  borderRadius: "8px",
                  border: "none",
                  background: !personName || isCapturing ? "#ccc" : "#28a745",
                  color: "white",
                  cursor: !personName || isCapturing ? "not-allowed" : "pointer",
                }}
              >
                Capture Image
              </button>
              <button
                onClick={() => startAutoCapture(20, 500)}
                disabled={isCapturing || !personName}
                style={{
                  padding: "12px 24px",
                  fontSize: "16px",
                  borderRadius: "8px",
                  border: "none",
                  background: isCapturing || !personName ? "#ccc" : "#007bff",
                  color: "white",
                  cursor: isCapturing || !personName ? "not-allowed" : "pointer",
                }}
              >
                {isCapturing ? "Capturing..." : "Auto Capture (20 images)"}
              </button>
            </div>
          ) : (
            <div style={{ marginTop: "20px" }}>
              <h3 style={{ color: "green" }}>
                ✅ Done capturing 20 images for{" "}
                <span style={{ textTransform: "capitalize" }}>{personName}</span>!
              </h3>
              <button
                onClick={resetForNewPerson}
                style={{
                  marginTop: "10px",
                  marginRight: "10px",
                  padding: "12px 24px",
                  backgroundColor: "#007bff",
                  color: "#fff",
                  border: "none",
                  borderRadius: "8px",
                  cursor: "pointer",
                  fontSize: "16px",
                }}
              >
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
                style={{
                  marginTop: "10px",
                  padding: "12px 24px",
                  backgroundColor: "#28a745",
                  color: "#fff",
                  border: "none",
                  borderRadius: "8px",
                  cursor: "pointer",
                  fontSize: "16px",
                }}
              >
                🗄️ Add Person to Database
              </button>
            </div>
          )}

          <h3 style={{ marginTop: "30px" }}>
            Captured Images ({capturedImages.length})
          </h3>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, 120px)",
              gap: "10px",
              justifyContent: "center",
              padding: "20px",
            }}
          >
            {capturedImages.map((img, i) => (
              <img
                key={i}
                src={img}
                alt={`capture-${i}`}
                style={{ width: "120px", borderRadius: "8px", border: "2px solid #ddd" }}
              />
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