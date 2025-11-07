import React, { useState, useEffect } from "react";
import axios from "axios";
import "../styles/Recording.css";

function RecordingsList() {
  const [recordings, setRecordings] = useState([]);
  const [selectedRecording, setSelectedRecording] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchRecordings();
  }, []);

  const fetchRecordings = async () => {
    try {
      setLoading(true);
      const response = await axios.get("http://localhost:8080/api/recordings");
      setRecordings(response.data.recordings);
      setError(null);
    } catch (err) {
      setError("Failed to fetch recordings");
      console.error("Error fetching recordings:", err);
    } finally {
      setLoading(false);
    }
  };

  const fetchRecordingDetails = async (filename) => {
    try {
      const response = await axios.get(`http://localhost:8080/api/recordings/${filename}`);
      setSelectedRecording(response.data.recording);
    } catch (err) {
      console.error("Error fetching recording details:", err);
      alert("Failed to fetch recording details");
    }
  };

  const deleteRecording = async (filename) => {
    if (!confirm(`Are you sure you want to delete "${filename}"?`)) return;

    try {
      await axios.delete(`http://localhost:8080/api/recordings/${filename}`);
      setRecordings(recordings.filter(r => r.filename !== filename));
      if (selectedRecording && selectedRecording.filename === filename) {
        setSelectedRecording(null);
      }
      alert("Recording deleted successfully");
    } catch (err) {
      console.error("Error deleting recording:", err);
      alert("Failed to delete recording");
    }
  };

  const formatDate = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  if (loading) {
    return (
      <div className="recording-container">
        <h2>📋 Recordings List</h2>
        <div className="loading">Loading recordings...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="recording-container">
        <h2>📋 Recordings List</h2>
        <div className="error">{error}</div>
        <button onClick={fetchRecordings} className="btn">Retry</button>
      </div>
    );
  }

  return (
    <div className="recording-container">
      <h2 className="recording-title">📋 Recordings List</h2>

      <div className="recordings-grid">
        <div className="recordings-list">
          <h3>Available Recordings ({recordings.length})</h3>
          {recordings.length === 0 ? (
            <p className="no-recordings">No recordings found</p>
          ) : (
            <div className="recordings-items">
              {recordings.map((recording) => (
                <div
                  key={recording.filename}
                  className={`recording-item ${selectedRecording?.filename === recording.filename ? 'selected' : ''}`}
                  onClick={() => fetchRecordingDetails(recording.filename)}
                >
                  <div className="recording-header">
                    <strong>{recording.filename}</strong>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteRecording(recording.filename);
                      }}
                      className="delete-btn"
                      title="Delete recording"
                    >
                      🗑️
                    </button>
                  </div>
                  <div className="recording-meta">
                    <div>📅 {formatDate(recording.timestamp)}</div>
                    <div>⏱️ {recording.actual_duration}s</div>
                    <div>📏 {recording.file_size_mb} MB</div>
                    <div>👥 {recording.total_unique_persons} persons</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="recording-details">
          {selectedRecording ? (
            <div className="details-card">
              <h3>📋 Recording Details</h3>
              <div className="details-content">
                <div className="detail-row">
                  <strong>Filename:</strong> {selectedRecording.filename}
                </div>
                <div className="detail-row">
                  <strong>Timestamp:</strong> {formatDate(selectedRecording.timestamp)}
                </div>
                <div className="detail-row">
                  <strong>Duration:</strong> {selectedRecording.actual_duration} seconds
                </div>
                <div className="detail-row">
                  <strong>FPS:</strong> {selectedRecording.actual_fps}
                </div>
                <div className="detail-row">
                  <strong>File Size:</strong> {selectedRecording.file_size_mb} MB
                </div>
                <div className="detail-row">
                  <strong>Resolution:</strong> {selectedRecording.resolution}
                </div>
                <div className="detail-row">
                  <strong>Codec:</strong> {selectedRecording.codec}
                </div>
                <div className="detail-row">
                  <strong>Threshold:</strong> {selectedRecording.threshold}
                </div>
                <div className="detail-row">
                  <strong>Total Unique Persons:</strong> {selectedRecording.total_unique_persons}
                </div>
                <div className="detail-row">
                  <strong>Detected Persons:</strong>
                  <div className="persons-list">
                    {selectedRecording.detected_persons.length > 0
                      ? selectedRecording.detected_persons.join(", ")
                      : "None"
                    }
                  </div>
                </div>
                {selectedRecording.person_times && selectedRecording.person_times.length > 0 && (
                  <div className="detail-row">
                    <strong>Person Times:</strong>
                    <div className="person-times">
                      {selectedRecording.person_times.map((pt, i) => (
                        <div key={i} className="person-time">
                          <span className="person-name">{pt.name}:</span>
                          <span>{pt.time}s</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
              <button
                onClick={() => deleteRecording(selectedRecording.filename)}
                className="btn delete-btn-large"
              >
                🗑️ Delete Recording
              </button>
            </div>
          ) : (
            <div className="no-selection">
              <div className="no-selection-icon">📋</div>
              <h3>Select a recording</h3>
              <p>Click on a recording from the list to view its details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default RecordingsList;