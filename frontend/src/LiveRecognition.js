import React, { useState, useEffect, useRef } from "react";
import axios from "axios";

function LiveRecognition() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [threshold, setThreshold] = useState(0.6);
  const [fps, setFps] = useState(15);
  const [databaseInfo, setDatabaseInfo] = useState(null);
  const [streamUrl, setStreamUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  
  const imgRef = useRef(null);

  // Fetch database info on mount
  useEffect(() => {
    fetchDatabaseInfo();
  }, []);

  const fetchDatabaseInfo = async () => {
    try {
      const response = await axios.get("http://192.168.1.213:8000/database_info");
      setDatabaseInfo(response.data);
    } catch (err) {
      console.error("Failed to fetch database info:", err);
    }
  };

  const startStream = () => {
    setIsLoading(true);
    const url = `http://192.168.1.213:8000/video_feed?threshold=${threshold}&fps=${fps}&t=${Date.now()}`;
    setStreamUrl(url);
    setIsStreaming(true);
    setTimeout(() => setIsLoading(false), 1000);
  };

  const stopStream = async () => {
    try {
      // Call backend to properly close camera
      await axios.post("http://192.168.1.213:8000/stop_stream");
      console.log("✅ Camera stream stopped on backend");
    } catch (err) {
      console.error("Failed to stop stream on backend:", err);
    }
    
    // Clear frontend stream
    setStreamUrl("");
    setIsStreaming(false);
    setIsLoading(false);
  };

  const handleThresholdChange = (e) => {
    const newThreshold = parseFloat(e.target.value);
    setThreshold(newThreshold);
    
    if (isStreaming) {
      const url = `http://192.168.1.213:8000/video_feed?threshold=${newThreshold}&fps=${fps}&t=${Date.now()}`;
      setStreamUrl(url);
    }
  };

  const handleFpsChange = (e) => {
    const newFps = parseInt(e.target.value);
    setFps(newFps);
    
    if (isStreaming) {
      const url = `http://192.168.1.213:8000/video_feed?threshold=${threshold}&fps=${newFps}&t=${Date.now()}`;
      setStreamUrl(url);
    }
  };

  const reloadDatabase = async () => {
    try {
      await axios.post("http://192.168.1.213:8000/reload_database");
      await fetchDatabaseInfo();
      alert("✅ Database reloaded successfully!");
    } catch (err) {
      alert("❌ Failed to reload database: " + err.message);
    }
  };

  return (
    <div style={{ padding: "20px", maxWidth: "1200px", margin: "0 auto" }}>
      <h2 style={{ textAlign: "center", color: "#333" }}>
        🎥 Live Face Recognition Stream
      </h2>

      {/* Database Info Card */}
      {databaseInfo && (
        <div
          style={{
            background: "#f0f8ff",
            padding: "15px",
            borderRadius: "8px",
            marginBottom: "20px",
            border: "1px solid #b0d4f1",
          }}
        >
          <h3 style={{ margin: "0 0 10px 0", color: "#0066cc" }}>
            📊 Database Status
          </h3>
          <div style={{ display: "flex", gap: "20px", flexWrap: "wrap" }}>
            <div>
              <strong>Users Loaded:</strong> {databaseInfo.users_count}
            </div>
            <div>
              <strong>Using InsightFace:</strong>{" "}
              {databaseInfo.using_insightface ? "✅ Yes" : "⚠️ No (Haar)"}
            </div>
            <div>
              <strong>Database Exists:</strong>{" "}
              {databaseInfo.database_exists ? "✅ Yes" : "❌ No"}
            </div>
          </div>
          {databaseInfo.users_count > 0 && (
            <div style={{ marginTop: "10px" }}>
              <strong>Registered Users:</strong>{" "}
              {Object.keys(databaseInfo.users).join(", ")}
            </div>
          )}
          <button
            onClick={reloadDatabase}
            style={{
              marginTop: "10px",
              padding: "8px 16px",
              backgroundColor: "#0066cc",
              color: "white",
              border: "none",
              borderRadius: "5px",
              cursor: "pointer",
            }}
          >
            🔄 Reload Database
          </button>
        </div>
      )}

      {/* Controls Panel */}
      <div
        style={{
          background: "#ffffff",
          padding: "20px",
          borderRadius: "8px",
          marginBottom: "20px",
          border: "1px solid #ddd",
          boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
        }}
      >
        <h3 style={{ marginTop: 0 }}>⚙️ Stream Controls</h3>

        {/* Threshold Slider */}
        <div style={{ marginBottom: "20px" }}>
          <label style={{ display: "block", marginBottom: "8px", fontWeight: "bold" }}>
            🎯 Recognition Threshold: {threshold.toFixed(2)}
          </label>
          <input
            type="range"
            min="0.3"
            max="0.9"
            step="0.05"
            value={threshold}
            onChange={handleThresholdChange}
            style={{ width: "100%", cursor: "pointer" }}
          />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", color: "#666" }}>
            <span>0.3 (More lenient)</span>
            <span>0.9 (More strict)</span>
          </div>
        </div>

        {/* FPS Slider */}
        <div style={{ marginBottom: "20px" }}>
          <label style={{ display: "block", marginBottom: "8px", fontWeight: "bold" }}>
            🎬 Frame Rate: {fps} FPS
          </label>
          <input
            type="range"
            min="5"
            max="30"
            step="5"
            value={fps}
            onChange={handleFpsChange}
            style={{ width: "100%", cursor: "pointer" }}
          />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", color: "#666" }}>
            <span>5 FPS (Lower CPU)</span>
            <span>30 FPS (Smoother)</span>
          </div>
        </div>

        {/* Start/Stop Buttons */}
        <div style={{ display: "flex", gap: "10px" }}>
          {!isStreaming ? (
            <button
              onClick={startStream}
              disabled={isLoading || (databaseInfo && databaseInfo.users_count === 0)}
              style={{
                padding: "12px 24px",
                backgroundColor: databaseInfo && databaseInfo.users_count === 0 ? "#ccc" : "#28a745",
                color: "white",
                border: "none",
                borderRadius: "8px",
                cursor: databaseInfo && databaseInfo.users_count === 0 ? "not-allowed" : "pointer",
                fontSize: "16px",
                fontWeight: "bold",
              }}
            >
              {isLoading ? "⏳ Starting..." : "▶️ Start Stream"}
            </button>
          ) : (
            <button
              onClick={stopStream}
              style={{
                padding: "12px 24px",
                backgroundColor: "#dc3545",
                color: "white",
                border: "none",
                borderRadius: "8px",
                cursor: "pointer",
                fontSize: "16px",
                fontWeight: "bold",
              }}
            >
              ⏹️ Stop Stream
            </button>
          )}
        </div>

        {databaseInfo && databaseInfo.users_count === 0 && (
          <p style={{ color: "#dc3545", marginTop: "10px", marginBottom: 0 }}>
            ⚠️ No users in database. Please capture and build database first.
          </p>
        )}
      </div>

      {/* Video Stream Display */}
      <div
        style={{
          background: "#000",
          borderRadius: "8px",
          overflow: "hidden",
          position: "relative",
          minHeight: "400px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {isStreaming ? (
          <img
            ref={imgRef}
            src={streamUrl}
            alt="Live Recognition Feed"
            style={{
              width: "100%",
              height: "auto",
              display: "block",
            }}
            onError={() => {
              console.error("Stream failed to load");
              alert("❌ Failed to connect to camera stream. Check Raspberry Pi connection.");
              stopStream();
            }}
          />
        ) : (
          <div style={{ textAlign: "center", color: "#666" }}>
            <div style={{ fontSize: "64px", marginBottom: "20px" }}>📹</div>
            <p style={{ fontSize: "18px", margin: 0 }}>
              {isLoading ? "Connecting to camera..." : "Click 'Start Stream' to begin"}
            </p>
          </div>
        )}
      </div>

      {/* Instructions */}
      <div
        style={{
          marginTop: "20px",
          padding: "15px",
          background: "#fff9e6",
          borderRadius: "8px",
          border: "1px solid #ffe066",
        }}
      >
        <h4 style={{ marginTop: 0 }}>💡 Tips:</h4>
        <ul style={{ marginBottom: 0, paddingLeft: "20px" }}>
          <li>
            <strong>Green boxes</strong> = Recognized faces
          </li>
          <li>
            <strong>Red boxes</strong> = Unknown faces
          </li>
          <li>
            Adjust threshold if recognition is too strict/lenient
          </li>
          <li>
            Lower FPS if stream is laggy (reduces CPU load)
          </li>
          <li>
            You can change threshold and FPS in real-time while streaming
          </li>
          <li>
            Make sure you're on the same network as the Raspberry Pi
          </li>
          <li>
            To record videos, use the "Recording" tab
          </li>
        </ul>
      </div>
    </div>
  );
}

export default LiveRecognition;