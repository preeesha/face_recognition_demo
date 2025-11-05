import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import "../styles/LiveRecognition.css";

function LiveRecognition() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [threshold, setThreshold] = useState(0.6);
  const [fps, setFps] = useState(15);
  const [databaseInfo, setDatabaseInfo] = useState(null);
  const [streamUrl, setStreamUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  
  const imgRef = useRef(null);

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
      await axios.post("http://192.168.1.213:8000/stop_stream");
      console.log("✅ Camera stream stopped on backend");
    } catch (err) {
      console.error("Failed to stop stream on backend:", err);
    }
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
    <div className="live-container">
      <h2 className="live-title">🎥 Live Face Recognition Stream</h2>

      {databaseInfo && (
        <div className="database-card">
          <h3 className="database-title">📊 Database Status</h3>
          <div className="database-info">
            <div><strong>Users Loaded:</strong> {databaseInfo.users_count}</div>
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
            <div className="registered-users">
              <strong>Registered Users:</strong>{" "}
              {Object.keys(databaseInfo.users).join(", ")}
            </div>
          )}
          <button onClick={reloadDatabase} className="reload-btn">
            🔄 Reload Database
          </button>
        </div>
      )}

      <div className="controls-panel">
        <h3>⚙️ Stream Controls</h3>

        <div className="slider-section">
          <label>
            🎯 Recognition Threshold: {threshold.toFixed(2)}
          </label>
          <input
            type="range"
            min="0.3"
            max="0.9"
            step="0.05"
            value={threshold}
            onChange={handleThresholdChange}
          />
          <div className="slider-labels">
            <span>0.3 (More lenient)</span>
            <span>0.9 (More strict)</span>
          </div>
        </div>

        <div className="slider-section">
          <label>
            🎬 Frame Rate: {fps} FPS
          </label>
          <input
            type="range"
            min="5"
            max="30"
            step="5"
            value={fps}
            onChange={handleFpsChange}
          />
          <div className="slider-labels">
            <span>5 FPS (Lower CPU)</span>
            <span>30 FPS (Smoother)</span>
          </div>
        </div>

        <div className="button-group">
          {!isStreaming ? (
            <button
              onClick={startStream}
              disabled={isLoading || (databaseInfo && databaseInfo.users_count === 0)}
              className={`start-btn ${databaseInfo && databaseInfo.users_count === 0 ? "disabled" : ""}`}
            >
              {isLoading ? "⏳ Starting..." : "▶️ Start Stream"}
            </button>
          ) : (
            <button onClick={stopStream} className="stop-btn">
              ⏹️ Stop Stream
            </button>
          )}
        </div>

        {databaseInfo && databaseInfo.users_count === 0 && (
          <p className="warning-text">
            ⚠️ No users in database. Please capture and build database first.
          </p>
        )}
      </div>

      <div className="video-container">
        {isStreaming ? (
          <img
            ref={imgRef}
            src={streamUrl}
            alt="Live Recognition Feed"
            onError={() => {
              console.error("Stream failed to load");
              alert("❌ Failed to connect to camera stream. Check Raspberry Pi connection.");
              stopStream();
            }}
          />
        ) : (
          <div className="no-stream">
            <div className="no-stream-icon">📹</div>
            <p>{isLoading ? "Connecting to camera..." : "Click 'Start Stream' to begin"}</p>
          </div>
        )}
      </div>

      <div className="tips-card">
        <h4>💡 Tips:</h4>
        <ul>
          <li><strong>Green boxes</strong> = Recognized faces</li>
          <li><strong>Red boxes</strong> = Unknown faces</li>
          <li>Adjust threshold if recognition is too strict/lenient</li>
          <li>Lower FPS if stream is laggy (reduces CPU load)</li>
          <li>You can change threshold and FPS in real-time while streaming</li>
          <li>Make sure you're on the same network as the Raspberry Pi</li>
          <li>To record videos, use the "Recording" tab</li>
        </ul>
      </div>
    </div>
  );
}

export default LiveRecognition;
