"""
Copyright (c) 2012 Bas Stottelaar, Jeroen Senden
See the file LICENSE for copying permission.
"""

from utils import GeneratorLen

import sqlite3
import cPickle
import cStringIO
import gzip
import threading

class Database(object):

    def __init__(self, path):
        self.lock = threading.Lock()
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute("CREATE TABLE IF NOT EXISTS faces (filename TEXT, person_id TEXT, data BLOB);")

    def flush(self):
        with self.lock:
            self.connection.execute("DELETE FROM faces;")
            self.connection.commit()

    def exists(self, filename):
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute("SELECT filename FROM faces WHERE filename = ? LIMIT 1", (filename, ))
        
        for row in cursor:
            return True

        return False

    def load(self, filename):
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute("SELECT filename, data FROM faces WHERE filename = ? LIMIT 1", (filename, ))

        for filename, data in cursor:
            string_file = cStringIO.StringIO(data)
            return cPickle.load(gzip.GzipFile(fileobj=string_file, mode='rb'))

    def save(self, filename, person_id, data):
        # Serialize and compress data
        string_file = cStringIO.StringIO()

        with gzip.GzipFile(fileobj=string_file, mode='wb') as gzip_file:
            gzip_file.write(cPickle.dumps(data))

        # Create SQLite compatible data
        data = sqlite3.Binary(string_file.getvalue())

        # Insert data and commit
        with self.lock:
            self.connection.execute("INSERT INTO faces(filename, person_id, data) VALUES (?, ?, ?)", (filename, person_id, data))
            self.connection.commit()
        
    def iterator(self):
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute("SELECT filename, person_id FROM faces ORDER BY person_id ASC")
            rows = [ (filename, person_id) for filename, person_id, in cursor ]
            count = len(rows)

        def _generator():
            for (filename, person_id) in rows:
                yield (filename, person_id, self.load(filename))

        return GeneratorLen(_generator(), count)