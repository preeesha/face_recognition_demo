import express from "express";
import { uploadRecording, upload, getRecordingsList, getRecordingDetails, deleteRecording } from "../controllers/recordingController.js";

const router = express.Router();
router.post("/upload_recording", upload.single("file"), uploadRecording);
router.get("/recordings", getRecordingsList);
router.get("/recordings/:filename", getRecordingDetails);
router.delete("/recordings/:filename", deleteRecording);

export default router;
