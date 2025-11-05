import React, { useState, useEffect, useRef } from "react";
import axios from "axios";

function Recording() {
  const [threshold, setThreshold] = useState(0.6);
  const [databaseInfo, setDatabaseInfo] = useState(null);

  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(60);
  const [recordingStatus, setRecordingStatus] = useState(null);
  const [streamUrl, setStreamUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const recordingIntervalRef = useRef(null);
  const imgRef = useRef(null);

  // Fetch database info on mount
  useEffect(() => {
    fetchDatabaseInfo();
  }, []);

  // Poll recording status when recording
  useEffect(() => {
    if (isRecording) {
      recordingIntervalRef.current = setInterval(async () => {
        try {
          const response = await axios.get(
            "http://192.168.1.213:8000/recording_status"
          );
          setRecordingStatus(response.data);

          // Check if recording finished AND upload complete
          if (!response.data.is_recording && response.data.upload_complete) {
            setIsRecording(false);
            clearInterval(recordingIntervalRef.current);
            
            // Show appropriate alert based on upload status
            if (response.data.upload_status === "success") {
              alert("✅ Recording completed and uploaded successfully!");
            } else if (response.data.upload_status === "failed") {
              alert("⚠️ Recording completed but upload failed!");
            }
            
            // Keep stream running - don't stop it
            // User can manually stop if needed
          }
        } catch (err) {
          console.error("Failed to fetch recording status:", err);
        }
      }, 1000);
    } else {
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
      }
    }

    return () => {
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
      }
    };
  }, [isRecording]);

  const fetchDatabaseInfo = async () => {
    try {
      const response = await axios.get(
        "http://192.168.1.213:8000/database_info"
      );
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

      // Start the video stream first
      const url = `http://192.168.1.213:8000/video_feed?threshold=${threshold}&fps=15&t=${Date.now()}`;
      setStreamUrl(url);

      // Wait for stream to start
      await new Promise((resolve) => setTimeout(resolve, 1000));

      // Then start recording
      const response = await axios.post(
        "http://192.168.1.213:8000/start_recording",
        null,
        {
          params: {
            duration: recordingDuration,
            threshold: threshold,
            fps: 4,
          },
        }
      );

      if (response.data.status === "started") {
        setIsRecording(true);
        setIsLoading(false);
      } else {
        throw new Error(response.data.message || "Failed to start recording");
      }
    } catch (err) {
      setIsLoading(false);
      setStreamUrl(""); // Clear stream on error
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

  const stopRecording = async () => {
    try {
      // Just stop the recording flag
      setIsRecording(false);

      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
      }

      alert("⚠️ Recording stopped early! Video may still be uploading.");
    } catch (err) {
      console.error("Failed to stop recording:", err);
    }
  };

  const handleThresholdChange = (e) => {
    const newThreshold = parseFloat(e.target.value);
    setThreshold(newThreshold);
  };

  return (
    <div style={{ padding: "20px", maxWidth: "1200px", margin: "0 auto" }}>
      <h2 style={{ textAlign: "center", color: "#333" }}>
        🎬 Recording Session
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

      {/* Recording Controls Panel */}
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
        <h3 style={{ marginTop: 0 }}>⚙️ Recording Settings</h3>

        {/* Recording Duration Slider */}
        <div style={{ marginBottom: "20px" }}>
          <label
            style={{
              display: "block",
              marginBottom: "8px",
              fontWeight: "bold",
            }}
          >
            ⏱️ Recording Duration: {recordingDuration} seconds
          </label>
          <input
            type="range"
            min="10"
            max="300"
            step="10"
            value={recordingDuration}
            onChange={(e) => setRecordingDuration(parseInt(e.target.value))}
            disabled={isRecording}
            style={{
              width: "100%",
              cursor: isRecording ? "not-allowed" : "pointer",
            }}
          />
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: "12px",
              color: "#666",
            }}
          >
            <span>10 seconds</span>
            <span>5 minutes</span>
          </div>
        </div>

        {/* Threshold Slider */}
        <div style={{ marginBottom: "20px" }}>
          <label
            style={{
              display: "block",
              marginBottom: "8px",
              fontWeight: "bold",
            }}
          >
            🎯 Recognition Threshold: {threshold.toFixed(2)}
          </label>
          <input
            type="range"
            min="0.3"
            max="0.9"
            step="0.05"
            value={threshold}
            onChange={handleThresholdChange}
            disabled={isRecording}
            style={{
              width: "100%",
              cursor: isRecording ? "not-allowed" : "pointer",
            }}
          />
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: "12px",
              color: "#666",
            }}
          >
            <span>0.3 (More lenient)</span>
            <span>0.9 (More strict)</span>
          </div>
        </div>

        {/* Recording Status Display */}
        {recordingStatus && recordingStatus.is_recording && (
          <div
            style={{
              background: "#fff3cd",
              padding: "15px",
              borderRadius: "8px",
              marginBottom: "15px",
              border: "2px solid #ffc107",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "10px",
                marginBottom: "10px",
              }}
            >
              <div
                style={{
                  width: "12px",
                  height: "12px",
                  borderRadius: "50%",
                  backgroundColor: "#dc3545",
                  animation: "pulse 1s infinite",
                }}
              ></div>
              <strong style={{ color: "#dc3545", fontSize: "18px" }}>
                🔴 RECORDING IN PROGRESS
              </strong>
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "10px",
                fontSize: "14px",
              }}
            >
              <div>
                <strong>Filename:</strong>
                <div
                  style={{ fontSize: "12px", color: "#666", marginTop: "2px" }}
                >
                  {recordingStatus.filename}
                </div>
              </div>
              <div>
                <strong>Progress:</strong>
                <div
                  style={{ fontSize: "12px", color: "#666", marginTop: "2px" }}
                >
                  {recordingStatus.elapsed_seconds}s / {recordingDuration}s
                </div>
              </div>
              <div>
                <strong>Frames Captured:</strong>
                <div
                  style={{ fontSize: "12px", color: "#666", marginTop: "2px" }}
                >
                  {recordingStatus.frame_count} frames
                </div>
              </div>
              <div>
                <strong>Detected Persons:</strong>
                <div
                  style={{ fontSize: "12px", color: "#666", marginTop: "2px" }}
                >
                  {(recordingStatus.detected_persons ?? []).length > 0
                    ? recordingStatus.detected_persons.join(", ")
                    : "None detected yet"}
                </div>
              </div>
            </div>

            {/* Progress Bar */}
            <div style={{ marginTop: "15px" }}>
              <div
                style={{
                  width: "100%",
                  height: "8px",
                  backgroundColor: "#e0e0e0",
                  borderRadius: "4px",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${
                      (recordingStatus.elapsed_seconds / recordingDuration) *
                      100
                    }%`,
                    height: "100%",
                    backgroundColor: "#dc3545",
                    transition: "width 0.3s ease",
                  }}
                ></div>
              </div>
            </div>
          </div>
        )}

        {/* Upload Status Indicator */}
        {!isRecording && recordingStatus && recordingStatus.upload_status === "uploading" && (
          <div
            style={{
              background: "#fff3cd",
              padding: "15px",
              borderRadius: "8px",
              marginBottom: "15px",
              border: "2px solid #ffc107",
              textAlign: "center",
            }}
          >
            <strong>📤 Uploading recording to server...</strong>
          </div>
        )}

        {/* Start/Stop Recording Button */}
        <div style={{ display: "flex", gap: "10px" }}>
          {!isRecording ? (
            <button
              onClick={startRecording}
              disabled={
                isLoading || (databaseInfo && databaseInfo.users_count === 0)
              }
              style={{
                padding: "15px 30px",
                backgroundColor: isLoading
                  ? "#6c757d"
                  : databaseInfo && databaseInfo.users_count === 0
                  ? "#ccc"
                  : "#dc3545",
                color: "white",
                border: "none",
                borderRadius: "8px",
                cursor:
                  isLoading || (databaseInfo && databaseInfo.users_count === 0)
                    ? "not-allowed"
                    : "pointer",
                fontSize: "18px",
                fontWeight: "bold",
                width: "100%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "10px",
              }}
            >
              <span style={{ fontSize: "24px" }}>⏺️</span>
              {isLoading ? "Starting..." : "Start Recording"}
            </button>
          ) : (
            <button
              onClick={stopRecording}
              style={{
                padding: "15px 30px",
                backgroundColor: "#6c757d",
                color: "white",
                border: "none",
                borderRadius: "8px",
                cursor: "pointer",
                fontSize: "18px",
                fontWeight: "bold",
                width: "100%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "10px",
              }}
            >
              <span style={{ fontSize: "24px" }}>⏹️</span>
              Stop Recording Early
            </button>
          )}
          
          {/* Stop Stream Button (when stream is active but not recording) */}
          {streamUrl && !isRecording && !isLoading && (
            <button
              onClick={stopStream}
              style={{
                padding: "15px 30px",
                backgroundColor: "#6c757d",
                color: "white",
                border: "none",
                borderRadius: "8px",
                cursor: "pointer",
                fontSize: "18px",
                fontWeight: "bold",
                width: "100%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "10px",
              }}
            >
              <span style={{ fontSize: "24px" }}>🛑</span>
              Stop Stream
            </button>
          )}
        </div>

        {databaseInfo && databaseInfo.users_count === 0 && (
          <p
            style={{
              color: "#dc3545",
              marginTop: "15px",
              marginBottom: 0,
              textAlign: "center",
            }}
          >
            ⚠️ No users in database. Please capture faces and build database
            first.
          </p>
        )}
      </div>

      {/* Camera Feed Display */}
      <div
        style={{
          background: "#000",
          borderRadius: "8px",
          overflow: "hidden",
          position: "relative",
          minHeight: "500px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          border: isRecording ? "3px solid #dc3545" : "1px solid #ddd",
        }}
      >
        {streamUrl ? (
          <div style={{ position: "relative", width: "100%" }}>
            <img
              ref={imgRef}
              src={streamUrl}
              alt="Recording Feed"
              style={{
                width: "100%",
                height: "auto",
                display: "block",
              }}
              onLoad={() => {
                console.log("✅ Stream loaded successfully");
              }}
              onError={(e) => {
                console.error("Stream failed to load", e);
                alert(
                  "❌ Failed to connect to camera stream. Check Raspberry Pi connection."
                );
                setStreamUrl("");
                setIsLoading(false);
              }}
            />

            {/* Recording Indicator Overlay */}
            {isRecording && recordingStatus && (
              <>
                <div
                  style={{
                    position: "absolute",
                    top: "20px",
                    left: "20px",
                    background: "rgba(220, 53, 69, 0.9)",
                    padding: "10px 20px",
                    borderRadius: "8px",
                    color: "white",
                    fontWeight: "bold",
                    fontSize: "16px",
                    display: "flex",
                    alignItems: "center",
                    gap: "10px",
                    boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
                  }}
                >
                  <div
                    style={{
                      width: "12px",
                      height: "12px",
                      borderRadius: "50%",
                      backgroundColor: "white",
                      animation: "pulse 1s infinite",
                    }}
                  ></div>
                  REC {recordingStatus?.elapsed_seconds || 0}s
                </div>

                {/* Detected Persons Overlay */}
                {recordingStatus &&
                  (recordingStatus.detected_persons ?? []).length > 0 && (
                    <div
                      style={{
                        position: "absolute",
                        top: "20px",
                        right: "20px",
                        background: "rgba(40, 167, 69, 0.9)",
                        padding: "10px 15px",
                        borderRadius: "8px",
                        color: "white",
                        fontSize: "14px",
                        boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
                      }}
                    >
                      <div style={{ fontWeight: "bold", marginBottom: "5px" }}>
                        👥 Detected (
                        {(recordingStatus.detected_persons ?? []).length})
                      </div>
                      {(recordingStatus.detected_persons ?? []).map(
                        (person, idx) => (
                          <div key={idx} style={{ fontSize: "12px" }}>
                            ✓ {person}
                          </div>
                        )
                      )}
                    </div>
                  )}
              </>
            )}
          </div>
        ) : (
          <div style={{ textAlign: "center", color: "#666", padding: "40px" }}>
            <div style={{ fontSize: "80px", marginBottom: "20px" }}>🎬</div>
            <h3 style={{ margin: "0 0 10px 0" }}>Ready to Record</h3>
            <p style={{ fontSize: "16px", margin: "0", color: "#999" }}>
              Configure settings above and click 'Start Recording' to begin
            </p>
            <div style={{ marginTop: "20px", fontSize: "14px", color: "#666" }}>
              <p>The camera feed will appear here during recording</p>
              <p>with live face recognition and detection overlays</p>
            </div>
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
        <h4 style={{ marginTop: 0 }}>💡 Recording Tips:</h4>
        <ul style={{ marginBottom: 0, paddingLeft: "20px" }}>
          <li>
            Recording automatically saves to Raspberry Pi and uploads to Express
            server
          </li>
          <li>
            <strong>Green boxes</strong> = Recognized faces |{" "}
            <strong>Red boxes</strong> = Unknown faces
          </li>
          <li>
            The camera feed shows live during recording with face recognition
            overlay
          </li>
          <li>
            Adjust threshold and duration before starting (cannot change during
            recording)
          </li>
          <li>Stream stays active after recording completes</li>
          <li>Recording stops automatically after the set duration</li>
          <li>
            You can stop recording early by clicking "Stop Recording Early"
          </li>
          <li>Use "Stop Stream" button to close the camera feed</li>
          <li>Make sure you're on the same network as the Raspberry Pi</li>
        </ul>
      </div>

      <style>
        {`
          @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
          }
        `}
      </style>
    </div>
  );
}

export default Recording;