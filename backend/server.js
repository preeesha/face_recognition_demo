// const express = require("express");
// const fs = require("fs");
// const path = require("path");
// const cors = require("cors");
// const archiver = require("archiver");
// const axios = require("axios");
// const FormData = require("form-data");
// const multer = require("multer");

// const { calculatePersonTimes } = require("./helper");

// const app = express();
// app.use(cors());
// app.use(express.json({ limit: "10mb" }));

// const DATASET_DIR = path.join(__dirname, "dataset");
// const RECORDINGS_DIR = path.join(__dirname, "uploaded_recordings");
// const db = require("./database");

// if (!fs.existsSync(DATASET_DIR)) fs.mkdirSync(DATASET_DIR, { recursive: true });
// if (!fs.existsSync(RECORDINGS_DIR))
//   fs.mkdirSync(RECORDINGS_DIR, { recursive: true });

// // =========================================
// // 🧠 1️⃣ Upload dataset images
// // =========================================
// app.post("/upload", (req, res) => {
//   try {
//     const { name, image, index } = req.body;
//     if (!name || !image) return res.status(400).send("Missing data");

//     const personDir = path.join(DATASET_DIR, name);
//     if (!fs.existsSync(personDir)) fs.mkdirSync(personDir, { recursive: true });

//     const imgData = image.replace(/^data:image\/jpeg;base64,/, "");
//     const filePath = path.join(personDir, `${name}_${index}.jpg`);
//     fs.writeFileSync(filePath, imgData, "base64");

//     console.log(`✅ Saved image: ${filePath}`);
//     res.json({ status: "ok" });
//   } catch (err) {
//     console.error("❌ Upload failed:", err.message);
//     res.status(500).json({ error: err.message });
//   }
// });

// // =========================================
// // ⚙️ 2️⃣ Helper: zip & send dataset to Raspberry Pi
// // =========================================
// async function sendDatasetToPi(personName) {
//   const personDir = path.join(DATASET_DIR, personName);
//   const zipPath = path.join(DATASET_DIR, `${personName}.zip`);

//   if (!fs.existsSync(personDir)) {
//     throw new Error(`Person directory not found: ${personDir}`);
//   }

//   // Create zip of the person's dataset
//   const output = fs.createWriteStream(zipPath);
//   const archive = archiver("zip", { zlib: { level: 9 } });

//   return new Promise((resolve, reject) => {
//     output.on("close", async () => {
//       console.log(`📦 Zipped ${archive.pointer()} bytes for ${personName}`);

//       const formData = new FormData();
//       formData.append("person_name", personName);
//       formData.append("file", fs.createReadStream(zipPath));

//       try {
//         const response = await axios.post(
//           "http://192.168.1.213:8000/upload_dataset",
//           formData,
//           {
//             headers: formData.getHeaders(),
//             maxContentLength: Infinity,
//             maxBodyLength: Infinity,
//           }
//         );

//         console.log("✅ Dataset sent to Pi:", response.data);
//         resolve(response.data);
//       } catch (err) {
//         console.error("❌ Failed to send dataset:", err.message);
//         reject(err);
//       }
//     });

//     archive.on("error", (err) => reject(err));
//     archive.pipe(output);
//     archive.directory(personDir, false);
//     archive.finalize();
//   });
// }

// // =========================================
// // 🚀 3️⃣ Endpoint to send dataset to Raspberry Pi
// // =========================================
// app.post("/send-to-pi", async (req, res) => {
//   const { name } = req.body;
//   if (!name) return res.status(400).json({ error: "Missing person name" });

//   try {
//     const result = await sendDatasetToPi(name);
//     res.json({ status: "sent", person: name, result });
//   } catch (err) {
//     console.error("❌ send-to-pi failed:", err.message);
//     res.status(500).json({ error: err.message });
//   }
// });

// // =========================================
// // 🎥 4️⃣ Receive recording uploads from Raspberry Pi
// // =========================================
// const storage = multer.diskStorage({
//   destination: (req, file, cb) => cb(null, RECORDINGS_DIR),
//   filename: (req, file, cb) => cb(null, file.originalname),
// });
// const upload = multer({ storage });

// app.post("/upload_recording", upload.single("file"), (req, res) => {
//   try {
//     if (!req.file) return res.status(400).json({ error: "No file uploaded" });

//     // Parse metadata with error handling
//     let metadata = {};
//     try {
//       metadata = req.body.metadata ? JSON.parse(req.body.metadata) : {};
//     } catch (err) {
//       console.error("⚠️ Failed to parse metadata:", err);
//       metadata = {};
//     }

//     // Add file size if not present
//     if (!metadata.file_size_mb && req.file.size) {
//       metadata.file_size_mb = parseFloat(
//         (req.file.size / (1024 * 1024)).toFixed(2)
//       );
//     }

//     console.log("\n🎬 New Recording Received!");
//     console.log("───────────────────────────────");
//     console.log(`📁 File saved at: ${req.file.path}`);
//     console.log(`📏 File size: ${metadata.file_size_mb} MB`);
//     console.log(`🕒 Timestamp: ${metadata.timestamp || "N/A"}`);
//     console.log(`⏱️ Duration: ${metadata.actual_duration || 0}s`);
//     console.log(`🎯 FPS: ${metadata.actual_fps || 0}`);
//     console.log(
//       `👥 Total Unique Persons: ${metadata.total_unique_persons || 0}`
//     );
//     console.log(
//       `🧠 Detected Persons: ${metadata.detected_persons?.join(", ") || "None"}`
//     );
//     console.log("───────────────────────────────\n");

//     // Save metadata to JSON file
//     const metaFilename = req.file.filename.replace(".avi", "_metadata.json");
//     const metaPath = path.join(RECORDINGS_DIR, metaFilename);

//     // Compute person presence times
//     const personTimes = calculatePersonTimes(metadata);
//     metadata.person_times = personTimes;

//     fs.writeFileSync(metaPath, JSON.stringify(metadata, null, 2));
//     console.log(`💾 Metadata saved: ${metaFilename}`);

//     try {
//       const insert = db.prepare(`
//       INSERT OR REPLACE INTO recordings 
//       (filename, filepath, file_size_mb, timestamp, actual_duration, actual_fps, total_unique_persons, detected_persons,person_times, resolution, codec, threshold)
//       VALUES (@filename, @filepath, @file_size_mb, @timestamp, @actual_duration, @actual_fps, @total_unique_persons, @detected_persons,@person_times, @resolution, @codec, @threshold)
//     `);

//       insert.run({
//         filename: req.file.filename,
//         filepath: req.file.path,
//         file_size_mb: metadata.file_size_mb || 0,
//         timestamp: metadata.timestamp || new Date().toISOString(),
//         actual_duration: metadata.actual_duration || 0,
//         actual_fps: metadata.actual_fps || 0,
//         total_unique_persons: metadata.total_unique_persons || 0,
//         detected_persons: JSON.stringify(metadata.detected_persons || []),
//         person_times: JSON.stringify(metadata.person_times || []),
//         resolution: metadata.resolution || "640x480",
//         codec: metadata.codec || "XVID",
//         threshold: metadata.threshold || 0.6,
//       });

//       console.log("✅ Metadata stored in SQLite DB:", req.file.filename);
//     } catch (dbErr) {
//       console.error("❌ Failed to insert metadata into SQLite:", dbErr.message);
//     }

//     res.status(200).json({
//       status: "ok",
//       message: "Recording received successfully",
//       filename: req.file.filename,
//       size_mb: metadata.file_size_mb,
//     });
//   } catch (err) {
//     console.error("❌ Error receiving recording:", err);
//     res.status(500).json({ error: err.message });
//   }
// });

// // =========================================
// // 🗂️ 5️⃣ View and manage uploaded recordings
// // =========================================
// app.get("/recordings", (req, res) => {
//   try {
//     const files = fs
//       .readdirSync(RECORDINGS_DIR)
//       .filter((f) => f.endsWith(".avi"));

//     const list = files.map((filename) => {
//       const metaFilename = filename.replace(".avi", "_metadata.json");
//       const metaPath = path.join(RECORDINGS_DIR, metaFilename);

//       // Default metadata
//       let metadata = {
//         filename: filename,
//         filepath: path.join(RECORDINGS_DIR, filename),
//         timestamp: new Date().toISOString(),
//         actual_duration: 0,
//         actual_fps: 0,
//         frame_count: 0,
//         file_size_mb: 0,
//         threshold: 0.6,
//         detected_persons: [],
//         person_detection_counts: {},
//         total_unique_persons: 0,
//         resolution: "640x480",
//         codec: "XVID",
//       };

//       // Load actual metadata from JSON file
//       if (fs.existsSync(metaPath)) {
//         try {
//           const metaContent = JSON.parse(fs.readFileSync(metaPath, "utf-8"));
//           metadata = { ...metadata, ...metaContent };
//         } catch (err) {
//           console.error(
//             `⚠️ Failed to load metadata for ${filename}:`,
//             err.message
//           );
//         }
//       } else {
//         console.warn(`⚠️ No metadata file found for ${filename}`);
//       }

//       return metadata;
//     });

//     res.json({ count: list.length, recordings: list });
//   } catch (err) {
//     console.error("❌ Error listing recordings:", err);
//     res.status(500).json({ error: err.message });
//   }
// });

// app.get("/recording/:filename", (req, res) => {
//   const filePath = path.join(RECORDINGS_DIR, req.params.filename);
//   if (!fs.existsSync(filePath)) {
//     return res.status(404).json({ error: "Recording not found" });
//   }
//   res.download(filePath);
// });

// // NEW: Delete recording endpoint
// app.delete("/recording/:filename", (req, res) => {
//   try {
//     const filename = req.params.filename;
//     const videoPath = path.join(RECORDINGS_DIR, filename);
//     const metaPath = path.join(
//       RECORDINGS_DIR,
//       filename.replace(".avi", "_metadata.json")
//     );

//     let deleted = [];

//     // Delete video file
//     if (fs.existsSync(videoPath)) {
//       fs.unlinkSync(videoPath);
//       deleted.push(videoPath);
//       console.log(`🗑️ Deleted video: ${filename}`);
//     }

//     // Delete metadata file
//     if (fs.existsSync(metaPath)) {
//       fs.unlinkSync(metaPath);
//       deleted.push(metaPath);
//       console.log(
//         `🗑️ Deleted metadata: ${filename.replace(".avi", "_metadata.json")}`
//       );
//     }

//     if (deleted.length === 0) {
//       return res.status(404).json({ error: "Recording not found" });
//     }

//     res.json({
//       status: "ok",
//       message: "Recording deleted successfully",
//       deleted: deleted,
//     });
//   } catch (err) {
//     console.error("❌ Delete failed:", err);
//     res.status(500).json({ error: err.message });
//   }
// });

// // =========================================
// // 🧩 6️⃣ Health Check
// // =========================================
// app.get("/", (req, res) => {
//   res.json({
//     status: "ok",
//     message: "✅ Node backend is running and ready for datasets + recordings!",
//     endpoints: {
//       upload: "POST /upload",
//       send_to_pi: "POST /send-to-pi",
//       upload_recording: "POST /upload_recording",
//       list_recordings: "GET /recordings",
//       download_recording: "GET /recording/:filename",
//       delete_recording: "DELETE /recording/:filename",
//     },
//   });
// });

// // =========================================
// // 🚀 Start Server
// // =========================================
// app.listen(8080, () => {
//   console.log("\nExpress server started");
//   console.log(`\n📍 Server running on http://localhost:8080`);
//   console.log(`📁 Dataset directory: ${DATASET_DIR}`);
//   console.log(`🎥 Recordings directory: ${RECORDINGS_DIR}`);
//   console.log(`\n📥 Waiting for dataset uploads and recordings...\n`);
// });

// import express from "express";
// import fs from "fs";
// import path from "path";
// import cors from "cors";
// import archiver from "archiver";
// import axios from "axios";
// import FormData from "form-data";
// import multer from "multer";

// import { fileURLToPath } from "url";
// import { dirname } from "path";
// import { calculatePersonTimes } from "./utils/helper.js";
// import db from "./config/database.js";

// // For __dirname in ESM
// const __filename = fileURLToPath(import.meta.url);
// const __dirname = dirname(__filename);

// const app = express();
// app.use(cors());
// app.use(express.json({ limit: "10mb" }));

// const DATASET_DIR = path.join(__dirname, "dataset");
// const RECORDINGS_DIR = path.join(__dirname, "uploaded_recordings");

// if (!fs.existsSync(DATASET_DIR)) fs.mkdirSync(DATASET_DIR, { recursive: true });
// if (!fs.existsSync(RECORDINGS_DIR))
//   fs.mkdirSync(RECORDINGS_DIR, { recursive: true });

// // =========================================
// // 🧠 1️⃣ Upload dataset images
// // =========================================
// app.post("/upload", (req, res) => {
//   try {
//     const { name, image, index } = req.body;
//     if (!name || !image) return res.status(400).send("Missing data");

//     const personDir = path.join(DATASET_DIR, name);
//     if (!fs.existsSync(personDir)) fs.mkdirSync(personDir, { recursive: true });

//     const imgData = image.replace(/^data:image\/jpeg;base64,/, "");
//     const filePath = path.join(personDir, `${name}_${index}.jpg`);
//     fs.writeFileSync(filePath, imgData, "base64");

//     console.log(`✅ Saved image: ${filePath}`);
//     res.json({ status: "ok" });
//   } catch (err) {
//     console.error("❌ Upload failed:", err.message);
//     res.status(500).json({ error: err.message });
//   }
// });

// // =========================================
// // ⚙️ 2️⃣ Helper: zip & send dataset to Raspberry Pi
// // =========================================
// async function sendDatasetToPi(personName) {
//   const personDir = path.join(DATASET_DIR, personName);
//   const zipPath = path.join(DATASET_DIR, `${personName}.zip`);

//   if (!fs.existsSync(personDir)) {
//     throw new Error(`Person directory not found: ${personDir}`);
//   }

//   const output = fs.createWriteStream(zipPath);
//   const archive = archiver("zip", { zlib: { level: 9 } });

//   return new Promise((resolve, reject) => {
//     output.on("close", async () => {
//       console.log(`📦 Zipped ${archive.pointer()} bytes for ${personName}`);

//       const formData = new FormData();
//       formData.append("person_name", personName);
//       formData.append("file", fs.createReadStream(zipPath));

//       try {
//         const response = await axios.post(
//           "http://192.168.1.213:8000/upload_dataset",
//           formData,
//           {
//             headers: formData.getHeaders(),
//             maxContentLength: Infinity,
//             maxBodyLength: Infinity,
//           }
//         );

//         console.log("✅ Dataset sent to Pi:", response.data);
//         resolve(response.data);
//       } catch (err) {
//         console.error("❌ Failed to send dataset:", err.message);
//         reject(err);
//       }
//     });

//     archive.on("error", (err) => reject(err));
//     archive.pipe(output);
//     archive.directory(personDir, false);
//     archive.finalize();
//   });
// }

// // =========================================
// // 🚀 3️⃣ Endpoint to send dataset to Raspberry Pi
// // =========================================
// app.post("/send-to-pi", async (req, res) => {
//   const { name } = req.body;
//   if (!name) return res.status(400).json({ error: "Missing person name" });

//   try {
//     const result = await sendDatasetToPi(name);
//     res.json({ status: "sent", person: name, result });
//   } catch (err) {
//     console.error("❌ send-to-pi failed:", err.message);
//     res.status(500).json({ error: err.message });
//   }
// });

// // =========================================
// // 🎥 4️⃣ Receive recording uploads from Raspberry Pi
// // =========================================
// const storage = multer.diskStorage({
//   destination: (req, file, cb) => cb(null, RECORDINGS_DIR),
//   filename: (req, file, cb) => cb(null, file.originalname),
// });
// const upload = multer({ storage });

// app.post("/upload_recording", upload.single("file"), (req, res) => {
//   try {
//     if (!req.file) return res.status(400).json({ error: "No file uploaded" });

//     let metadata = {};
//     try {
//       metadata = req.body.metadata ? JSON.parse(req.body.metadata) : {};
//     } catch (err) {
//       console.error("⚠️ Failed to parse metadata:", err);
//       metadata = {};
//     }

//     if (!metadata.file_size_mb && req.file.size) {
//       metadata.file_size_mb = parseFloat(
//         (req.file.size / (1024 * 1024)).toFixed(2)
//       );
//     }

//     console.log("\n🎬 New Recording Received!");
//     console.log("───────────────────────────────");
//     console.log(`📁 File saved at: ${req.file.path}`);
//     console.log(`📏 File size: ${metadata.file_size_mb} MB`);
//     console.log(`🕒 Timestamp: ${metadata.timestamp || "N/A"}`);
//     console.log(`⏱️ Duration: ${metadata.actual_duration || 0}s`);
//     console.log(`🎯 FPS: ${metadata.actual_fps || 0}`);
//     console.log(
//       `👥 Total Unique Persons: ${metadata.total_unique_persons || 0}`
//     );
//     console.log(
//       `🧠 Detected Persons: ${metadata.detected_persons?.join(", ") || "None"}`
//     );
//     console.log("───────────────────────────────\n");

//     const metaFilename = req.file.filename.replace(".avi", "_metadata.json");
//     const metaPath = path.join(RECORDINGS_DIR, metaFilename);

//     const personTimes = calculatePersonTimes(metadata);
//     metadata.person_times = personTimes;

//     fs.writeFileSync(metaPath, JSON.stringify(metadata, null, 2));
//     console.log(`💾 Metadata saved: ${metaFilename}`);

//     try {
//       const insert = db.prepare(`
//         INSERT OR REPLACE INTO recordings 
//         (filename, filepath, file_size_mb, timestamp, actual_duration, actual_fps, total_unique_persons, detected_persons, person_times, resolution, codec, threshold)
//         VALUES (@filename, @filepath, @file_size_mb, @timestamp, @actual_duration, @actual_fps, @total_unique_persons, @detected_persons, @person_times, @resolution, @codec, @threshold)
//       `);

//       insert.run({
//         filename: req.file.filename,
//         filepath: req.file.path,
//         file_size_mb: metadata.file_size_mb || 0,
//         timestamp: metadata.timestamp || new Date().toISOString(),
//         actual_duration: metadata.actual_duration || 0,
//         actual_fps: metadata.actual_fps || 0,
//         total_unique_persons: metadata.total_unique_persons || 0,
//         detected_persons: JSON.stringify(metadata.detected_persons || []),
//         person_times: JSON.stringify(metadata.person_times || []),
//         resolution: metadata.resolution || "640x480",
//         codec: metadata.codec || "XVID",
//         threshold: metadata.threshold || 0.6,
//       });

//       console.log("✅ Metadata stored in SQLite DB:", req.file.filename);
//     } catch (dbErr) {
//       console.error("❌ Failed to insert metadata into SQLite:", dbErr.message);
//     }

//     res.status(200).json({
//       status: "ok",
//       message: "Recording received successfully",
//       filename: req.file.filename,
//       size_mb: metadata.file_size_mb,
//     });
//   } catch (err) {
//     console.error("❌ Error receiving recording:", err);
//     res.status(500).json({ error: err.message });
//   }
// });

// // =========================================
// // 🗂️ 5️⃣ View and manage uploaded recordings
// // =========================================
// app.get("/recordings", (req, res) => {
//   try {
//     const files = fs
//       .readdirSync(RECORDINGS_DIR)
//       .filter((f) => f.endsWith(".avi"));

//     const list = files.map((filename) => {
//       const metaFilename = filename.replace(".avi", "_metadata.json");
//       const metaPath = path.join(RECORDINGS_DIR, metaFilename);

//       let metadata = {
//         filename,
//         filepath: path.join(RECORDINGS_DIR, filename),
//         timestamp: new Date().toISOString(),
//         actual_duration: 0,
//         actual_fps: 0,
//         frame_count: 0,
//         file_size_mb: 0,
//         threshold: 0.6,
//         detected_persons: [],
//         person_detection_counts: {},
//         total_unique_persons: 0,
//         resolution: "640x480",
//         codec: "XVID",
//       };

//       if (fs.existsSync(metaPath)) {
//         try {
//           const metaContent = JSON.parse(fs.readFileSync(metaPath, "utf-8"));
//           metadata = { ...metadata, ...metaContent };
//         } catch (err) {
//           console.error(`⚠️ Failed to load metadata for ${filename}:`, err.message);
//         }
//       } else {
//         console.warn(`⚠️ No metadata file found for ${filename}`);
//       }

//       return metadata;
//     });

//     res.json({ count: list.length, recordings: list });
//   } catch (err) {
//     console.error("❌ Error listing recordings:", err);
//     res.status(500).json({ error: err.message });
//   }
// });

// // =========================================
// // 🧩 Health Check
// // =========================================
// app.get("/", (req, res) => {
//   res.json({
//     status: "ok",
//     message: "✅ Node backend (ESM) running and ready!",
//   });
// });

// // =========================================
// // 🚀 Start Server
// // =========================================
// app.listen(8080, () => {
//   console.log("\nExpress server (ESM) started");
//   console.log(`📍 Running at http://localhost:8080`);
// });

import express from "express";
import cors from "cors";
import datasetRoutes from "./routes/datasetRoutes.js";
import recordingRoutes from "./routes/recordingRoutes.js";

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));

app.use("/api", datasetRoutes);
app.use("/api", recordingRoutes);

app.get("/", (req, res) => {
  res.json({ status: "ok", message: "Backend running perfectly!" });
});



app.listen(8080, "0.0.0.0", () => {
  console.log("🚀 Server running on http://0.0.0.0:8080");
});