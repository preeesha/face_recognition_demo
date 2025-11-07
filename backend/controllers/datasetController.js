import fs from "fs";
import path from "path";
import archiver from "archiver";
import axios from "axios";
import FormData from "form-data";
import db from "../config/database.js";
import { DATASET_DIR } from "../config/paths.js";

export async function uploadDataset(req, res) {
  try {
    const { name, image, index } = req.body;
    if (!name || !image) return res.status(400).send("Missing data");

    const personDir = path.join(DATASET_DIR, name);
    if (!fs.existsSync(personDir)) fs.mkdirSync(personDir, { recursive: true });

    const imgData = image.replace(/^data:image\/jpeg;base64,/, "");
    const filePath = path.join(personDir, `${name}_${index}.jpg`);
    fs.writeFileSync(filePath, imgData, "base64");

    console.log(`✅ Saved image: ${filePath}`);
    res.json({ status: "ok" });
  } catch (err) {
    console.error("❌ Upload failed:", err.message);
    res.status(500).json({ error: err.message });
  }
}

export async function sendDatasetToPi(req, res) {
  const { name } = req.body;
  if (!name) return res.status(400).json({ error: "Missing person name" });

  try {
    const personDir = path.join(DATASET_DIR, name);
    const zipPath = path.join(DATASET_DIR, `${name}.zip`);

    if (!fs.existsSync(personDir)) {
      throw new Error(`Person directory not found: ${personDir}`);
    }

    const output = fs.createWriteStream(zipPath);
    const archive = archiver("zip", { zlib: { level: 9 } });

    archive.directory(personDir, false);
    archive.pipe(output);
    archive.finalize();

    output.on("close", async () => {
      const formData = new FormData();
      formData.append("person_name", name);
      formData.append("file", fs.createReadStream(zipPath));

      const response = await axios.post(
        "http://192.168.1.213:8000/upload_dataset",
        formData,
        { headers: formData.getHeaders() }
      );

      console.log("✅ Dataset sent to Pi:", response.data);
      res.json({ status: "sent", name });
    });
  } catch (err) {
    console.error("❌ send-to-pi failed:", err.message);
    res.status(500).json({ error: err.message });
  }
}

export async function get_recordings(req, res){
  try {
    const rows = db.prepare("SELECT * FROM recordings ORDER BY timestamp DESC").all();
    res.json(rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
}