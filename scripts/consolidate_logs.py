"""
Consolidate all experiment logs into a single training_logs.txt file.

Usage:
    python scripts/consolidate_logs.py
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import LOGS_DIR


def consolidate_logs():
    """Merge all .log files into training_logs.txt."""
    output_file = LOGS_DIR / "training_logs.txt"
    log_files = sorted(LOGS_DIR.glob("*.log"))
    
    if not log_files:
        print("⚠️  No .log files found in logs/")
        return
    
    print(f"📝 Consolidating {len(log_files)} log files...")
    
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write(f"="*80 + "\n")
        outfile.write(f"CONSOLIDATED TRAINING LOGS\n")
        outfile.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        outfile.write(f"="*80 + "\n\n")
        
        for log_file in log_files:
            print(f"  • {log_file.name}")
            outfile.write(f"\n{'='*80}\n")
            outfile.write(f"FILE: {log_file.name}\n")
            outfile.write(f"{'='*80}\n\n")
            
            with open(log_file, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
            
            outfile.write(f"\n\n")
    
    print(f"\n✅ Consolidated log saved to: {output_file}")
    print(f"📊 Total size: {output_file.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    consolidate_logs()
