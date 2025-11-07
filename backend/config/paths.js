import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";
import fs from "fs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const DATASET_DIR = path.join(__dirname, "../dataset");
const RECORDINGS_DIR = path.join(__dirname, "../uploaded_recordings");

if (!fs.existsSync(DATASET_DIR)) fs.mkdirSync(DATASET_DIR, { recursive: true });
if (!fs.existsSync(RECORDINGS_DIR)) fs.mkdirSync(RECORDINGS_DIR, { recursive: true });

export { DATASET_DIR, RECORDINGS_DIR };
