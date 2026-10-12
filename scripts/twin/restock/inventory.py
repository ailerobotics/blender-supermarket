"""Transactional stock reservations and durable mission progress.

An interrupted physical action is never replayed automatically. Its reservation
remains held until an operator reconciles the observed stock and actuator state.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Job:
    job_id: str
    sku: str
    quantity: int
    source_bin: str
    shelf_id: str

    def __post_init__(self):
        if type(self.quantity) is not int or self.quantity <= 0:
            raise ValueError("quantity must be a positive integer")
        if not all(isinstance(v, str) and v.strip() for v in
                   (self.job_id, self.sku, self.source_bin, self.shelf_id)):
            raise ValueError("job, SKU, source and shelf IDs are required")
        if self.source_bin == self.shelf_id:
            raise ValueError("source and destination must differ")


class Inventory:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS stock (
                location TEXT, sku TEXT, quantity INTEGER NOT NULL CHECK(quantity>=0),
                PRIMARY KEY(location, sku));
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, spec TEXT NOT NULL, state TEXT NOT NULL,
                verified INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY, job_id TEXT NOT NULL,
                state TEXT NOT NULL, evidence TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS observations (
                observation_id TEXT PRIMARY KEY, job_id TEXT NOT NULL);
        ''')

    def close(self):
        self.db.close()

    def seed(self, location, sku, quantity):
        """Initial stock only; cannot overwrite an operational inventory."""
        if type(quantity) is not int or quantity < 0:
            raise ValueError("stock must be a nonnegative integer")
        with self.db:
            self.db.execute("INSERT INTO stock VALUES(?,?,?)", (location, sku, quantity))

    def count(self, location, sku):
        row = self.db.execute("SELECT quantity FROM stock WHERE location=? AND sku=?",
                              (location, sku)).fetchone()
        return row[0] if row else 0

    def reserve(self, job: Job):
        with self.db:
            # Acquire the write lock before inspecting other reservations.
            self.db.execute("BEGIN IMMEDIATE")
            if self.db.execute("SELECT 1 FROM jobs WHERE job_id=?", (job.job_id,)).fetchone():
                raise ValueError("existing job requires reconciliation; refusing replay")
            held = 0
            for row in self.db.execute("SELECT spec FROM jobs WHERE verified=0"):
                other = json.loads(row[0])
                if (other['source_bin'], other['sku']) == (job.source_bin, job.sku):
                    held += other['quantity']
            if self.count(job.source_bin, job.sku) - held < job.quantity:
                raise ValueError("insufficient unreserved stock")
            self.db.execute("INSERT INTO jobs(job_id,spec,state) VALUES(?,?,?)",
                            (job.job_id, json.dumps(asdict(job)), "reserved"))
            self._event(job.job_id, "reserved", {})

    def _event(self, job_id, state, evidence):
        self.db.execute("INSERT INTO events(job_id,state,evidence) VALUES(?,?,?)",
                        (job_id, state, json.dumps(evidence, allow_nan=False)))

    def record(self, job_id, state, evidence):
        with self.db:
            result = self.db.execute("UPDATE jobs SET state=? WHERE job_id=?", (state, job_id))
            if result.rowcount != 1:
                raise ValueError("unknown job")
            self._event(job_id, state, evidence)

    def verify(self, job: Job, evidence):
        """Commit a validated transfer exactly once; called by the mission gate."""
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute("SELECT spec,verified,state FROM jobs WHERE job_id=?",
                                  (job.job_id,)).fetchone()
            if not row or json.loads(row['spec']) != asdict(job):
                raise ValueError("job does not match reservation")
            if row['verified'] or row['state'] != 'verify':
                raise ValueError("job is not awaiting verification")
            self.db.execute("INSERT INTO observations VALUES(?,?)",
                            (evidence['observation_id'], job.job_id))
            result = self.db.execute(
                "UPDATE stock SET quantity=quantity-? WHERE location=? AND sku=? AND quantity>=?",
                (job.quantity, job.source_bin, job.sku, job.quantity))
            if result.rowcount != 1:
                raise ValueError("source stock no longer reconciles")
            self.db.execute('''INSERT INTO stock VALUES(?,?,?) ON CONFLICT(location,sku)
                               DO UPDATE SET quantity=quantity+excluded.quantity''',
                            (job.shelf_id, job.sku, job.quantity))
            self.db.execute("UPDATE jobs SET verified=1 WHERE job_id=?", (job.job_id,))
            self._event(job.job_id, 'inventory_verified', evidence)

    def events(self, job_id):
        return [dict(row) | {'evidence': json.loads(row['evidence'])}
                for row in self.db.execute("SELECT * FROM events WHERE job_id=? ORDER BY sequence",
                                           (job_id,))]
