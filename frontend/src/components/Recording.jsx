import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import "../styles/Recording.css";

function Recording() {
  const [threshold, setThreshold] = useState(0.6);
  const [databaseInfo, setDatabaseInfo] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(60);
  const [recordingStatus, setRecordingStatus] = useState(null);
  const [streamUrl, setStreamUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const recordingIntervalRef = useRef(null);
  const imgRef = useRef(null);

  useEffect(() => {
    fetchDatabaseInfo();
  }, []);

  useEffect(() => {
    if (isRecording) {
      recordingIntervalRef.current = setInterval(async () => {
        try {
          const response = await axios.get("http://192.168.1.213:8000/recording_status");
          setRecordingStatus(response.data);

          if (!response.data.is_recording && response.data.upload_complete) {
            setIsRecording(false);
            clearInterval(recordingIntervalRef.current);

            if (response.data.upload_status === "success") {
              alert("✅ Recording completed and uploaded successfully!");
            } else if (response.data.upload_status === "failed") {
              alert("⚠️ Recording completed but upload failed!");
            }
          }
        } catch (err) {
          console.error("Failed to fetch recording status:", err);
        }
      }, 1000);
    } else {
      if (recordingIntervalRef.current) clearInterval(recordingIntervalRef.current);
    }

    return () => {
      if (recordingIntervalRef.current) clearInterval(recordingIntervalRef.current);
    };
  }, [isRecording]);

  const fetchDatabaseInfo = async () => {
    try {
      const response = await axios.get("http://192.168.1.213:8000/database_info");
      setDatabaseInfo(response.data);
    } catch (err) {
      console.error("Failed to fetch database info:", err);
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

  const startRecording = async () => {
    try {
      setIsLoading(true);
      const url = `http://192.168.1.213:8000/video_feed?threshold=${threshold}&fps=15&t=${Date.now()}`;
      setStreamUrl(url);

      await new Promise((resolve) => setTimeout(resolve, 1000));

      const response = await axios.post("http://192.168.1.213:8000/start_recording", null, {
        params: { duration: recordingDuration, threshold, fps: 10 },
      });

      if (response.data.status === "started") {
        setIsRecording(true);
        setIsLoading(false);
      } else throw new Error(response.data.message || "Failed to start recording");
    } catch (err) {
      setIsLoading(false);
      setStreamUrl("");
      alert("❌ Failed to start recording: " + (err.response?.data?.message || err.message));
    }
  };

  const stopStream = async () => {
    try {
      await axios.post("http://192.168.1.213:8000/stop_stream");
      console.log("✅ Stream stopped");
    } catch (err) {
      console.error("Failed to stop stream:", err);
    }
    setStreamUrl("");
    setIsLoading(false);
  };

  const stopRecording = () => {
    setIsRecording(false);
    if (recordingIntervalRef.current) clearInterval(recordingIntervalRef.current);
    alert("⚠️ Recording stopped early! Video may still be uploading.");
  };

  const handleThresholdChange = (e) => setThreshold(parseFloat(e.target.value));

  return (
    <div className="recording-container">
      <h2 className="recording-title">🎬 Recording Session</h2>

      {databaseInfo && (
        <div className="database-card">
          <h3 className="database-title">📊 Database Status</h3>
          <div className="database-info">
            <div><strong>Users Loaded:</strong> {databaseInfo.users_count}</div>
            <div><strong>Using InsightFace:</strong> {databaseInfo.using_insightface ? "✅ Yes" : "⚠️ No (Haar)"}</div>
            <div><strong>Database Exists:</strong> {databaseInfo.database_exists ? "✅ Yes" : "❌ No"}</div>
          </div>
          {databaseInfo.users_count > 0 && (
            <div className="registered-users">
              <strong>Registered Users:</strong>{" "}
              {Object.keys(databaseInfo.users).join(", ")}
            </div>
          )}
          <button onClick={reloadDatabase} className="reload-btn">🔄 Reload Database</button>
        </div>
      )}

      <div className="controls-panel">
        <h3>⚙️ Recording Settings</h3>

        <div className="slider-section">
          <label>⏱️ Recording Duration: {recordingDuration} seconds</label>
          <input
            type="range"
            min="10"
            max="300"
            step="10"
            value={recordingDuration}
            onChange={(e) => setRecordingDuration(parseInt(e.target.value))}
            disabled={isRecording}
          />
          <div className="slider-labels">
            <span>10 seconds</span>
            <span>5 minutes</span>
          </div>
        </div>

        <div className="slider-section">
          <label>🎯 Recognition Threshold: {threshold.toFixed(2)}</label>
          <input
            type="range"
            min="0.3"
            max="0.9"
            step="0.05"
            value={threshold}
            onChange={handleThresholdChange}
            disabled={isRecording}
          />
          <div className="slider-labels">
            <span>0.3 (More lenient)</span>
            <span>0.9 (More strict)</span>
          </div>
        </div>

        {recordingStatus && recordingStatus.is_recording && (
          <div className="recording-status-card">
            <div className="recording-header">
              <div className="recording-dot"></div>
              <strong>🔴 RECORDING IN PROGRESS</strong>
            </div>
            <div className="status-grid">
              <div><strong>Filename:</strong><div>{recordingStatus.filename}</div></div>
              <div><strong>Progress:</strong><div>{recordingStatus.elapsed_seconds}s / {recordingDuration}s</div></div>
              <div><strong>Frames Captured:</strong><div>{recordingStatus.frame_count}</div></div>
              <div><strong>Detected Persons:</strong><div>{(recordingStatus.detected_persons ?? []).join(", ") || "None"}</div></div>
            </div>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${(recordingStatus.elapsed_seconds / recordingDuration) * 100}%` }}
              ></div>
            </div>
          </div>
        )}

        {!isRecording && recordingStatus && recordingStatus.upload_status === "uploading" && (
          <div className="upload-card">
            <strong>📤 Uploading recording to server...</strong>
          </div>
        )}

        <div className="button-group">
          {!isRecording ? (
            <button
              onClick={startRecording}
              disabled={isLoading || (databaseInfo && databaseInfo.users_count === 0)}
              className={`start-btn ${isLoading ? "loading" : ""}`}
            >
              <span>⏺️</span> {isLoading ? "Starting..." : "Start Recording"}
            </button>
          ) : (
            <button onClick={stopRecording} className="stop-btn">
              <span>⏹️</span> Stop Recording Early
            </button>
          )}

          {streamUrl && !isRecording && !isLoading && (
            <button onClick={stopStream} className="stop-stream-btn">
              <span>🛑</span> Stop Stream
            </button>
          )}
        </div>

        {databaseInfo && databaseInfo.users_count === 0 && (
          <p className="warning-text">
            ⚠️ No users in database. Please capture faces and build database first.
          </p>
        )}
      </div>

      <div className={`video-container ${isRecording ? "recording" : ""}`}>
        {streamUrl ? (
          <div className="video-wrapper">
            <img
              ref={imgRef}
              src={streamUrl}
              alt="Recording Feed"
              onError={() => {
                alert("❌ Failed to connect to camera stream. Check Raspberry Pi connection.");
                setStreamUrl("");
                setIsLoading(false);
              }}
            />
            {isRecording && recordingStatus && (
              <>
                <div className="rec-overlay">
                  <div className="rec-dot"></div> REC {recordingStatus?.elapsed_seconds || 0}s
                </div>
                {(recordingStatus.detected_persons ?? []).length > 0 && (
                  <div className="detected-overlay">
                    <div>👥 Detected ({(recordingStatus.detected_persons ?? []).length})</div>
                    {(recordingStatus.detected_persons ?? []).map((p, i) => (
                      <div key={i}>✓ {p}</div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        ) : (
          <div className="no-video">
            <div className="no-video-icon">🎬</div>
            <h3>Ready to Record</h3>
            <p>Configure settings above and click 'Start Recording' to begin</p>
          </div>
        )}
      </div>

      <div className="tips-card">
        <h4>💡 Recording Tips:</h4>
        <ul>
          <li>Recording automatically saves to Raspberry Pi and uploads to Express server</li>
          <li><strong>Green boxes</strong> = Recognized | <strong>Red boxes</strong> = Unknown</li>
          <li>Adjust threshold and duration before starting</li>
          <li>Stream stays active after recording completes</li>
          <li>You can stop recording early anytime</li>
        </ul>
      </div>
    </div>
  );
}

export default Recording;
