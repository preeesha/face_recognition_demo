import express from "express";
import { uploadRecording, upload } from "../controllers/recordingController.js";

const router = express.Router();
router.post("/upload_recording", upload.single("file"), uploadRecording);

export default router;
