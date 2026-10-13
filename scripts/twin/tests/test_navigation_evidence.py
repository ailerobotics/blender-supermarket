"""Behavioral tests for the navigation EvidenceSnapshot envelope consumer.

These exercise the pure-CPU mirror of the host typed IDL
(restock_mapper_evidence/EvidenceSnapshot). They assert what the consumer must
REFUSE; they do not stand in for native producer/runtime commissioning.
"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from twin.navigation.temporal import (MapIdentity,Acquisition,DistanceSlice,EvidenceEnvelope,
    TemporalMap,ADMISSION_OPEN,ADMISSION_CLOSED,ADMISSION_FAULT,
    EMPTY_POPULATED,EMPTY_VERIFIED,EMPTY_UNKNOWN)

DOMAIN='host-monotonic'


class EvidenceConsumerTests(unittest.TestCase):
    def setUp(self):
        self.clock=[1.]
        self.identity=MapIdentity('sim','layout','cam','loc')
        self.gate=TemporalMap(self.identity,commissioned=0.,required_sensors=['front','cart'],
            generation='empty-1',steady_clock=lambda:self.clock[0],steady_commissioned=0.,
            steady_domain=DOMAIN)
        self.distances=(2.,)*(240*160)

    def acq(self,sensor,stamp,*,domain=DOMAIN,frame=None):
        return Acquisition(sensor,self.identity,int(round(stamp*1000)),stamp,stamp,
                           optical_frame=frame or sensor+'_optical',steady_clock_domain=domain)

    def slice_for(self,acqs,*,generation=None,frame='map',incorporated=None):
        newest=incorporated or max(acqs,key=lambda a:a.stamp)
        return DistanceSlice(frame,(-6.,-4.),.05,240,160,-1000.,self.distances,
                             generation or self.gate.generation,newest)

    def envelope(self,acqs,*,snapshot_index=0,esdf_revision=0,nonce='N1',cert='C1',
                 admission_state=ADMISSION_OPEN,empty_state=EMPTY_POPULATED,domain=DOMAIN,
                 full_height=False,slice=None,completed=None):
        acqs=tuple(acqs)
        sl=slice if slice is not None else self.slice_for(acqs)
        completed=max(a.steady_stamp for a in acqs) if completed is None else completed
        return EvidenceEnvelope(sl,nonce,cert,self.identity,snapshot_index,esdf_revision,
                                acqs,completed,domain,admission_state,empty_state,full_height)

    def register(self,acqs):
        for a in acqs:self.gate.observe(a)
        return acqs

    def test_valid_envelope_is_accepted_and_republished(self):
        acqs=self.register([self.acq('front',1.),self.acq('cart',1.)])
        self.gate.accept_envelope(self.envelope(acqs))
        self.assertEqual(self.gate.readiness(1.),'')
        self.assertIs(self.gate.republished_slice(1.),self.gate.slice)
        self.assertEqual(set(self.gate.ledger),{'front','cart'})

    def test_incomplete_ledger_blocks_motion_and_republish(self):
        both=self.register([self.acq('front',1.),self.acq('cart',1.)])
        self.gate.accept_envelope(self.envelope([both[0]]))  # front only
        self.assertEqual(self.gate.readiness(1.),'incomplete_incorporation_ledger')
        self.assertIsNone(self.gate.republished_slice(1.))

    def test_stale_ledger_record_stops_even_when_live_cameras_fresh(self):
        old=self.register([self.acq('front',1.),self.acq('cart',1.)])
        self.gate.accept_envelope(self.envelope(old))
        self.register([self.acq('front',1.35),self.acq('cart',1.35)])  # fresh sources
        self.clock[0]=1.35
        self.assertEqual(self.gate.readiness(1.35),'stale_navigation_sensor_or_mapper')

    def test_admission_must_be_open(self):
        acqs=self.register([self.acq('front',1.),self.acq('cart',1.)])
        for state in (ADMISSION_CLOSED,ADMISSION_FAULT):
            with self.assertRaises(ValueError):
                self.gate.accept_envelope(self.envelope(acqs,admission_state=state))

    def test_only_populated_snapshot_certifies_free_space(self):
        acqs=self.register([self.acq('front',1.),self.acq('cart',1.)])
        for state in (EMPTY_UNKNOWN,EMPTY_VERIFIED):
            with self.assertRaises(ValueError):
                self.gate.accept_envelope(self.envelope(acqs,empty_state=state))

    def test_snapshot_index_must_strictly_increase(self):
        acqs=self.register([self.acq('front',1.),self.acq('cart',1.)])
        self.gate.accept_envelope(self.envelope(acqs,snapshot_index=3))
        fresh=self.register([self.acq('front',1.1),self.acq('cart',1.1)])
        for idx in (3,2):
            with self.assertRaises(ValueError):
                self.gate.accept_envelope(self.envelope(fresh,snapshot_index=idx,esdf_revision=1))

    def test_esdf_revision_may_repeat_but_never_decrease(self):
        acqs=self.register([self.acq('front',1.),self.acq('cart',1.)])
        self.gate.accept_envelope(self.envelope(acqs,snapshot_index=0,esdf_revision=5))
        again=self.register([self.acq('front',1.05),self.acq('cart',1.05)])
        self.gate.accept_envelope(self.envelope(again,snapshot_index=1,esdf_revision=5))  # repeat ok
        older=self.register([self.acq('front',1.1),self.acq('cart',1.1)])
        with self.assertRaises(ValueError):
            self.gate.accept_envelope(self.envelope(older,snapshot_index=2,esdf_revision=4))

    def test_producer_identity_is_one_shot_within_a_generation(self):
        acqs=self.register([self.acq('front',1.),self.acq('cart',1.)])
        self.gate.accept_envelope(self.envelope(acqs,nonce='N1',cert='C1'))
        fresh=self.register([self.acq('front',1.1),self.acq('cart',1.1)])
        for bad in (dict(nonce='N2'),dict(cert='C2'),dict(domain='other')):
            with self.assertRaises(ValueError):
                self.gate.accept_envelope(self.envelope(fresh,snapshot_index=1,**bad))

    def test_agreed_steady_domain_is_enforced(self):
        acqs=self.register([self.acq('front',1.,domain='wrong'),self.acq('cart',1.,domain='wrong')])
        with self.assertRaises(ValueError):
            self.gate.accept_envelope(self.envelope(acqs,domain='wrong'))

    def test_ledger_rejects_duplicate_and_unconfigured_sensors(self):
        acqs=self.register([self.acq('front',1.),self.acq('cart',1.)])
        with self.assertRaises(ValueError):
            self.gate.accept_envelope(self.envelope([acqs[0],acqs[0]]))
        rogue=Acquisition('lidar',self.identity,1000,1.,1.,optical_frame='lidar_optical',
                          steady_clock_domain=DOMAIN)
        # observe() refuses to register an unconfigured sensor at all.
        with self.assertRaises(ValueError):self.gate.observe(rogue)
        with self.assertRaises(ValueError):
            self.gate.accept_envelope(self.envelope([acqs[0],rogue]))

    def test_ledger_entry_must_be_registered_in_history(self):
        front=self.acq('front',1.);cart=self.acq('cart',1.)
        self.register([front])  # cart never registered
        with self.assertRaises(ValueError):
            self.gate.accept_envelope(self.envelope([front,cart]))

    def test_optical_frame_binding_is_immutable_within_a_generation(self):
        acqs=self.register([self.acq('front',1.),self.acq('cart',1.)])
        self.gate.accept_envelope(self.envelope(acqs))
        moved=self.register([self.acq('front',1.1,frame='front_moved'),self.acq('cart',1.1)])
        with self.assertRaises(ValueError):
            self.gate.accept_envelope(self.envelope(moved,snapshot_index=1))

    def test_embedded_slice_frame_generation_and_newest_binding(self):
        acqs=self.register([self.acq('front',1.1),self.acq('cart',1.)])
        with self.assertRaises(ValueError):
            self.gate.accept_envelope(self.envelope(acqs,slice=self.slice_for(acqs,generation='other')))
        with self.assertRaises(ValueError):
            self.gate.accept_envelope(self.envelope(acqs,slice=self.slice_for(acqs,frame='ur_base_link')))
        stale_bind=self.slice_for(acqs,incorporated=acqs[1])  # binds the older (cart@1.0)
        with self.assertRaises(ValueError):
            self.gate.accept_envelope(self.envelope(acqs,slice=stale_bind))

    def test_out_of_order_incorporation_rejected(self):
        older=self.register([self.acq('front',1.),self.acq('cart',1.)])
        newer=self.register([self.acq('front',1.1),self.acq('cart',1.1)])
        self.gate.accept_envelope(self.envelope(newer,snapshot_index=0))  # slice binds @1.1
        with self.assertRaises(ValueError):  # a later index but older incorporation
            self.gate.accept_envelope(self.envelope(older,snapshot_index=1))

    def test_full_height_gate_stays_closed_even_when_flag_true(self):
        acqs=self.register([self.acq('front',1.),self.acq('cart',1.)])
        self.gate.accept_envelope(self.envelope(acqs,full_height=True))
        self.assertTrue(self.gate.full_height_verified)   # flag recorded verbatim
        self.assertFalse(self.gate.full_height_certified())  # gate never opens here

    def test_new_empty_generation_requires_fresh_producer_identity(self):
        acqs=self.register([self.acq('front',1.),self.acq('cart',1.)])
        self.gate.accept_envelope(self.envelope(acqs,nonce='N1',cert='C1'))
        self.gate.invalidate_clearance(now=1.1)
        self.gate.begin_empty_reconstruction('empty-2')
        self.assertIsNone(self.gate.producer_nonce)
        self.assertEqual(self.gate.ledger,{})
        fresh=self.register([self.acq('front',1.2),self.acq('cart',1.2)])
        self.clock[0]=1.2
        self.gate.accept_envelope(self.envelope(fresh,nonce='N2',cert='C2'))
        self.assertEqual(self.gate.readiness(1.2),'')
        self.assertEqual(self.gate.producer_nonce,'N2')

    def test_rejected_first_envelope_leaves_producer_unbound(self):
        # A ledger-validation failure must not half-bind the producer identity.
        front=self.acq('front',1.);cart=self.acq('cart',1.)
        self.register([front])  # cart not registered -> acceptance will raise
        with self.assertRaises(ValueError):
            self.gate.accept_envelope(self.envelope([front,cart],nonce='N1',cert='C1'))
        self.assertIsNone(self.gate.producer_nonce)
        # The correct producer can still bind afterwards.
        self.register([cart])
        self.gate.accept_envelope(self.envelope([front,cart],nonce='N9',cert='C9'))
        self.assertEqual(self.gate.producer_nonce,'N9')

    def test_completion_delay_enters_the_braking_envelope(self):
        acqs=self.register([self.acq('front',1.),self.acq('cart',1.)])
        self.gate.accept_envelope(self.envelope(acqs,completed=1.))
        base=self.gate.reaction_delay(1.)
        self.gate.accept_envelope(self.envelope(
            self.register([self.acq('front',1.05),self.acq('cart',1.05)]),
            snapshot_index=1,completed=.5))  # producer completed 0.5s before now
        self.clock[0]=1.
        self.assertAlmostEqual(self.gate.reaction_delay(1.),.5)
        self.assertGreater(self.gate.reaction_delay(1.),base)


if __name__=='__main__':unittest.main()
