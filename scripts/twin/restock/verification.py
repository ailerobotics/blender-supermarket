"""Independent post-transfer inventory check using new rendered observations.

Starting verification discards transfer-phase frames. Both source vacancy and
destination identity must be scanned twice after that boundary. USD object poses,
controller completion, old commissioning planes and gripper release are not
inventory evidence.
"""
from dataclasses import asdict
import math


class InventoryVerifier:
    def __init__(self, capture, *, source='rollwagen'):
        self.capture,self.source=capture,source
        self.active=False

    def begin(self, job, now):
        if not math.isfinite(now):
            raise ValueError('finite verification boundary required')
        self.job,self.started=job,now
        self.last_stamp=now
        self.seen=set()
        self.confirmations=[]
        self.active=True

    def stop(self):
        self.active=False

    def __call__(self, now):
        if not self.active:
            return None
        packet=self.capture(now,self.source,self.job.shelf_id)
        if packet is None:
            return None
        stamp=packet.simulation_time
        if (not math.isfinite(stamp) or not 0<=now-stamp<=.5
                or not math.isfinite(packet.scan_time) or not 0<=now-packet.scan_time<=.5):
            raise ValueError('verification requires fresh inventory scans')
        if stamp<=self.last_stamp or packet.scan_time<=self.started:
            return None
        self.last_stamp=stamp
        if not packet.parked or packet.contact:
            raise ValueError('verification requires parked robot and released handling unit')
        if not {self.source,self.job.shelf_id}<=set(packet.scanned_locations):
            self.confirmations=[]
            return None
        source=[o for o in packet.observations if o.location==self.source]
        destination=[o for o in packet.observations if o.location==self.job.shelf_id]
        if source or len(destination)!=1:
            self.confirmations=[]
            return None
        observed=destination[0]
        if (observed.source!='sensor' or observed.sku!=self.job.sku
                or type(observed.quantity) is not int or observed.quantity!=self.job.quantity
                or observed.unit_kind!='handling_unit' or observed.frame!='ur_base_link'
                or not math.isfinite(observed.confidence) or observed.confidence<.9
                or not math.isfinite(observed.simulation_time)
                or not self.started<observed.simulation_time<=stamp
                or not 0<=now-observed.simulation_time<=.5
                or not observed.observation_id):
            raise ValueError('post-transfer identity or provenance mismatch')
        if observed.observation_id in self.seen:
            return None
        self.seen.add(observed.observation_id)
        self.confirmations.append(observed.observation_id)
        if len(self.confirmations)<2:
            return None
        self.active=False
        return dict(asdict(observed),transfer_confirmed=True,
                    source_vacancy_location=self.source,scan_time=packet.scan_time,
                    verification_started=self.started,
                    confirming_observation_ids=list(self.confirmations))
