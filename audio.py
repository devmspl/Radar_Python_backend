import sqlite3

conn = sqlite3.connect("test.db")
conn.execute("ALTER TABLE feeds ADD COLUMN source_type TEXT;")
conn.execute("ALTER TABLE feeds ADD COLUMN transcript_id TEXT;")
conn.commit()
conn.close()
