-- Database schema for DeepShield Forensic Engine

-- Analyses table
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash TEXT UNIQUE,
    file_type TEXT CHECK(file_type IN ('image', 'audio', 'video')),
    file_name TEXT,
    file_size INTEGER,
    result TEXT,
    confidence REAL,
    ensemble_confidence REAL,
    analysis_time REAL,
    features TEXT, -- JSON string
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_analyses_timestamp ON analyses(timestamp);
CREATE INDEX IF NOT EXISTS idx_analyses_file_type ON analyses(file_type);
CREATE INDEX IF NOT EXISTS idx_analyses_result ON analyses(result);

-- Detailed results table
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER,
    model_name TEXT,
    prediction REAL,
    confidence REAL,
    details TEXT, -- JSON string
    FOREIGN KEY (analysis_id) REFERENCES analyses (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_results_analysis_id ON results(analysis_id);

-- Statistics table
CREATE TABLE IF NOT EXISTS statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE UNIQUE,
    total_analyses INTEGER DEFAULT 0,
    deepfake_count INTEGER DEFAULT 0,
    real_count INTEGER DEFAULT 0,
    avg_confidence REAL DEFAULT 0,
    avg_analysis_time REAL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_statistics_date ON statistics(date);

-- Users table (for future authentication)
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    email TEXT UNIQUE,
    password_hash TEXT,
    api_key TEXT UNIQUE,
    role TEXT DEFAULT 'user',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME
);

-- API usage tracking
CREATE TABLE IF NOT EXISTS api_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    endpoint TEXT,
    requests_count INTEGER DEFAULT 1,
    last_used DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

-- Create views for common queries
CREATE VIEW IF NOT EXISTS vw_daily_stats AS
SELECT 
    date,
    total_analyses,
    deepfake_count,
    real_count,
    avg_confidence,
    avg_analysis_time,
    ROUND(100.0 * deepfake_count / NULLIF(total_analyses, 0), 2) as deepfake_percentage
FROM statistics
ORDER BY date DESC;

CREATE VIEW IF NOT EXISTS vw_model_performance AS
SELECT 
    model_name,
    COUNT(*) as predictions_count,
    AVG(prediction) as avg_prediction,
    AVG(confidence) as avg_confidence
FROM results
GROUP BY model_name;

-- Triggers
CREATE TRIGGER IF NOT EXISTS update_analysis_timestamp 
AFTER UPDATE ON analyses
BEGIN
    UPDATE analyses SET timestamp = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Sample data for testing
INSERT OR IGNORE INTO statistics (date, total_analyses, deepfake_count, real_count, avg_confidence, avg_analysis_time)
VALUES 
    (date('now', '-7 days'), 150, 45, 105, 0.82, 1.5),
    (date('now', '-6 days'), 175, 52, 123, 0.84, 1.4),
    (date('now', '-5 days'), 200, 68, 132, 0.81, 1.6),
    (date('now', '-4 days'), 190, 55, 135, 0.83, 1.5),
    (date('now', '-3 days'), 210, 72, 138, 0.85, 1.4),
    (date('now', '-2 days'), 225, 80, 145, 0.84, 1.5),
    (date('now', '-1 days'), 240, 85, 155, 0.86, 1.3),
    (date('now'), 180, 62, 118, 0.85, 1.4);