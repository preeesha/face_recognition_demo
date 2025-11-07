import fs from "fs";
import path from "path";
import multer from "multer";
import db from "../config/database.js";
import { RECORDINGS_DIR } from "../config/paths.js";
import { calculatePersonTimes } from "../utils/helper.js";

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, RECORDINGS_DIR),
  filename: (req, file, cb) => cb(null, file.originalname),
});
export const upload = multer({ storage });

export function uploadRecording(req, res) {
 try {
    if (!req.file) return res.status(400).json({ error: "No file uploaded" });

    let metadata = {};
    try {
      metadata = req.body.metadata ? JSON.parse(req.body.metadata) : {};
    } catch (err) {
      console.error("⚠️ Failed to parse metadata:", err);
      metadata = {};
    }

    if (!metadata.file_size_mb && req.file.size) {
      metadata.file_size_mb = parseFloat(
        (req.file.size / (1024 * 1024)).toFixed(2)
      );
    }

    console.log("\n🎬 New Recording Received!");
    console.log("───────────────────────────────");
    console.log(`📁 File saved at: ${req.file.path}`);
    console.log(`📏 File size: ${metadata.file_size_mb} MB`);
    console.log(`🕒 Timestamp: ${metadata.timestamp || "N/A"}`);
    console.log(`⏱️ Duration: ${metadata.actual_duration || 0}s`);
    console.log(`🎯 FPS: ${metadata.actual_fps || 0}`);
    console.log(
      `👥 Total Unique Persons: ${metadata.total_unique_persons || 0}`
    );
    console.log(
      `🧠 Detected Persons: ${metadata.detected_persons?.join(", ") || "None"}`
    );
    console.log("───────────────────────────────\n");

    const metaFilename = req.file.filename.replace(".avi", "_metadata.json");
    const metaPath = path.join(RECORDINGS_DIR, metaFilename);

    const personTimes = calculatePersonTimes(metadata);
    metadata.person_times = personTimes;

    fs.writeFileSync(metaPath, JSON.stringify(metadata, null, 2));
    console.log(`💾 Metadata saved: ${metaFilename}`);

    try {
      const insert = db.prepare(`
        INSERT OR REPLACE INTO recordings 
        (filename, filepath, file_size_mb, timestamp, actual_duration, actual_fps, total_unique_persons, detected_persons, person_times, resolution, codec, threshold)
        VALUES (@filename, @filepath, @file_size_mb, @timestamp, @actual_duration, @actual_fps, @total_unique_persons, @detected_persons, @person_times, @resolution, @codec, @threshold)
      `);

      insert.run({
        filename: req.file.filename,
        filepath: req.file.path,
        file_size_mb: metadata.file_size_mb || 0,
        timestamp: metadata.timestamp || new Date().toISOString(),
        actual_duration: metadata.actual_duration || 0,
        actual_fps: metadata.actual_fps || 0,
        total_unique_persons: metadata.total_unique_persons || 0,
        detected_persons: JSON.stringify(metadata.detected_persons || []),
        person_times: JSON.stringify(metadata.person_times || []),
        resolution: metadata.resolution || "640x480",
        codec: metadata.codec || "XVID",
        threshold: metadata.threshold || 0.6,
      });

      console.log("✅ Metadata stored in SQLite DB:", req.file.filename);
    } catch (dbErr) {
      console.error("❌ Failed to insert metadata into SQLite:", dbErr.message);
    }

    res.status(200).json({
      status: "ok",
      message: "Recording received successfully",
      filename: req.file.filename,
      size_mb: metadata.file_size_mb,
    });
  } catch (err) {
    console.error("❌ Error receiving recording:", err);
    res.status(500).json({ error: err.message });
  }
}


