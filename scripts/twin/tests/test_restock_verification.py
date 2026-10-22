from dataclasses import replace

import pytest

from twin.restock.inventory import Job
from twin.restock.verification import InventoryVerifier
from twin.restock_manipulation.types import Observation,Sensors


JOB=Job(job_id='j',sku='juice',quantity=1,source_bin='backroom',shelf_id='shelf')


def packet(stamp, identity='frame'):
    observed=Observation(identity,'sensor',stamp,'juice','shelf',(0,0,0,0,0,0,1),.99)
    return Sensors(stamp,(observed,),None,True,False,stamp,stamp,(0,)*6,
                   ('rollwagen','shelf'),stamp)


def verifier():
    current=[None]
    result=InventoryVerifier(lambda *args:current[0])
    result.begin(JOB,1.)
    return result,current


def test_new_independent_two_frame_confirmation():
    verify,current=verifier()
    current[0]=packet(.99,'transfer-frame')
    assert verify(1.1) is None
    current[0]=packet(1.1,'one')
    assert verify(1.1) is None
    current[0]=packet(1.2,'one')
    assert verify(1.2) is None
    current[0]=packet(1.3,'two')
    result=verify(1.3)
    assert result['transfer_confirmed'] is True
    assert result['confirming_observation_ids']==['one','two']
    assert verify(1.4) is None


@pytest.mark.parametrize('missing',[('shelf',),('rollwagen',),()])
def test_absence_of_a_source_detection_does_not_prove_vacancy(missing):
    verify,current=verifier()
    current[0]=replace(packet(1.1),scanned_locations=missing)
    assert verify(1.1) is None


def test_source_still_occupied_resets_confirmation():
    verify,current=verifier()
    current[0]=packet(1.1,'one');assert verify(1.1) is None
    occupied=packet(1.2,'two')
    current[0]=replace(occupied,observations=occupied.observations+
                       (replace(occupied.observations[0],location='rollwagen'),))
    assert verify(1.2) is None
    current[0]=packet(1.3,'three');assert verify(1.3) is None
    current[0]=packet(1.4,'four');assert verify(1.4)['transfer_confirmed']


@pytest.mark.parametrize('field,value',[('source','test_fixture'),('sku','other'),
                                        ('confidence',.5),('quantity',2)])
def test_reject_wrong_stock_and_non_sensor_evidence(field,value):
    verify,current=verifier()
    item=packet(1.1)
    current[0]=replace(item,observations=(replace(item.observations[0],**{field:value}),))
    with pytest.raises(ValueError,match='identity or provenance'):
        verify(1.1)
