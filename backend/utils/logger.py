import datetime
import json
import os
import sqlite3
from typing import Dict, List, Any
import pandas as pd

class ForensicLogger:
    """Log and manage analysis results"""
    
    def __init__(self, db_path='database/forensic.db'):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT UNIQUE,
                file_type TEXT,
                file_name TEXT,
                file_size INTEGER,
                result TEXT,
                confidence REAL,
                ensemble_confidence REAL,
                analysis_time REAL,
                features TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create detailed results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                model_name TEXT,
                prediction REAL,
                confidence REAL,
                details TEXT,
                FOREIGN KEY (analysis_id) REFERENCES analyses (id)
            )
        ''')
        
        # Create statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_analyses INTEGER DEFAULT 0,
                deepfake_count INTEGER DEFAULT 0,
                real_count INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0,
                avg_analysis_time REAL DEFAULT 0,
                date DATE UNIQUE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_analysis(self, file_type: str, file_hash: str, result: Dict, 
                    ensemble_result: Dict = None, file_name: str = "", 
                    file_size: int = 0):
        """Log analysis result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert main analysis
        cursor.execute('''
            INSERT OR REPLACE INTO analyses 
            (file_hash, file_type, file_name, file_size, result, confidence, 
             ensemble_confidence, analysis_time, features, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            file_hash,
            file_type,
            file_name,
            file_size,
            result.get('label', 'Unknown'),
            result.get('confidence', 0),
            ensemble_result.get('confidence', 0) if ensemble_result else None,
            result.get('analysis_time', 0),
            json.dumps(result.get('features', {})),
            datetime.datetime.now().isoformat()
        ))
        
        analysis_id = cursor.lastrowid
        
        # Insert detailed results
        if 'individual_predictions' in result:
            for model_name, prediction in result['individual_predictions'].items():
                cursor.execute('''
                    INSERT INTO results (analysis_id, model_name, prediction, details)
                    VALUES (?, ?, ?, ?)
                ''', (
                    analysis_id,
                    model_name,
                    prediction,
                    json.dumps({})
                ))
        
        # Update statistics
        self._update_statistics(result)
        
        conn.commit()
        conn.close()
        
        # Also log to text file for compatibility
        self._log_to_text(file_type, result)
    
    def _log_to_text(self, file_type, result):
        """Log to text file for backward compatibility"""
        with open("logs.txt", "a") as f:
            f.write(f"{datetime.datetime.now()} | {file_type} | {result.get('label', 'Unknown')} | {result.get('confidence', 0)}\n")
    
    def _update_statistics(self, result):
        """Update daily statistics"""
        today = datetime.datetime.now().date()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current stats for today
        cursor.execute('''
            SELECT total_analyses, deepfake_count, real_count, 
                   avg_confidence, avg_analysis_time
            FROM statistics WHERE date = ?
        ''', (today.isoformat(),))
        
        row = cursor.fetchone()
        
        if row:
            total, deepfake, real, avg_conf, avg_time = row
            total += 1
            
            if 'deepfake' in result.get('label', '').lower():
                deepfake += 1
            else:
                real += 1
            
            avg_conf = (avg_conf * (total - 1) + result.get('confidence', 0)) / total
            avg_time = (avg_time * (total - 1) + result.get('analysis_time', 0)) / total
            
            cursor.execute('''
                UPDATE statistics SET 
                    total_analyses = ?, deepfake_count = ?, real_count = ?,
                    avg_confidence = ?, avg_analysis_time = ?
                WHERE date = ?
            ''', (total, deepfake, real, avg_conf, avg_time, today.isoformat()))
        else:
            total = 1
            deepfake = 1 if 'deepfake' in result.get('label', '').lower() else 0
            real = 0 if deepfake else 1
            
            cursor.execute('''
                INSERT INTO statistics 
                (date, total_analyses, deepfake_count, real_count, 
                 avg_confidence, avg_analysis_time)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                today.isoformat(),
                total,
                deepfake,
                real,
                result.get('confidence', 0),
                result.get('analysis_time', 0)
            ))
        
        conn.commit()
        conn.close()
    
    def get_recent_reports(self, limit: int = 50) -> List[Dict]:
        """Get recent analysis reports"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM analyses 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        reports = [dict(row) for row in rows]
        
        conn.close()
        return reports
    
    def get_report(self, file_hash: str) -> Dict:
        """Get specific analysis report by file hash"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM analyses WHERE file_hash = ?
        ''', (file_hash,))
        
        row = cursor.fetchone()
        
        if row:
            report = dict(row)
            
            # Get detailed results
            cursor.execute('''
                SELECT * FROM results WHERE analysis_id = ?
            ''', (report['id'],))
            
            results = cursor.fetchall()
            report['detailed_results'] = [dict(r) for r in results]
            
            conn.close()
            return report
        
        conn.close()
        return None
    
    def get_statistics(self) -> Dict:
        """Get overall statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Overall stats
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN result LIKE '%Deepfake%' THEN 1 ELSE 0 END) as deepfakes,
                AVG(confidence) as avg_confidence,
                AVG(analysis_time) as avg_time
            FROM analyses
        ''')
        
        row = cursor.fetchone()
        total, deepfakes, avg_conf, avg_time = row
        
        # Today's stats
        today = datetime.datetime.now().date().isoformat()
        cursor.execute('''
            SELECT * FROM statistics WHERE date = ?
        ''', (today,))
        
        today_stats = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_analyses': total or 0,
            'deepfake_count': deepfakes or 0,
            'real_count': (total or 0) - (deepfakes or 0),
            'avg_confidence': float(avg_conf or 0),
            'avg_analysis_time': float(avg_time or 0),
            'accuracy': float(deepfakes or 0) / (total or 1) * 100,
            'today': {
                'date': today,
                'analyses': today_stats[1] if today_stats else 0,
                'deepfakes': today_stats[2] if today_stats else 0,
                'real': today_stats[3] if today_stats else 0
            } if today_stats else None
        }
    
    def export_to_csv(self, output_path: str = "reports.csv"):
        """Export all reports to CSV"""
        conn = sqlite3.connect(self.db_path)
        
        df = pd.read_sql_query("SELECT * FROM analyses ORDER BY timestamp DESC", conn)
        df.to_csv(output_path, index=False)
        
        conn.close()
        return output_path
    
    def clear_old_logs(self, days: int = 30):
        """Clear logs older than specified days"""
        cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM analyses WHERE timestamp < ?
        ''', (cutoff_date,))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted