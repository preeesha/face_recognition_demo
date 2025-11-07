import express from "express";
import { uploadDataset, sendDatasetToPi, get_recordings } from "../controllers/datasetController.js";

const router = express.Router();

router.post("/upload", uploadDataset);
router.post("/send-to-pi", sendDatasetToPi);



export default router;
