import Database from "better-sqlite3";
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const dbPath = path.join(__dirname, "recordings.db");
const db = new Database(dbPath);

db.prepare(`
  CREATE TABLE IF NOT EXISTS recordings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT UNIQUE,
    filepath TEXT,
    file_size_mb REAL,
    timestamp TEXT,
    actual_duration REAL,
    actual_fps REAL,
    total_unique_persons INTEGER,
    detected_persons TEXT,
    person_times TEXT,
    resolution TEXT,
    codec TEXT,
    threshold REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
  )
`).run();

console.log("📦 SQLite database initialized at:", dbPath);

export default db;
