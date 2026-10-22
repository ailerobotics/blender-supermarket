import xml.etree.ElementTree as ET
import pytest

from twin.restock.isaac_bootstrap import resolve_meshes


def test_mesh_resolution_preserves_original_and_real_geometry(tmp_path):
    root=tmp_path/'packages'
    mesh=root/'actual_robot'/'collision.stl'
    mesh.parent.mkdir(parents=True)
    mesh.write_text('fixture geometry')
    urdf=tmp_path/'original.urdf'
    original='<robot name="test"><link name="l"><collision><geometry><mesh filename="package://actual_robot/collision.stl"/></geometry></collision></link></robot>'
    urdf.write_text(original)
    output=resolve_meshes(urdf,tmp_path/'resolved.urdf',[str(root)])
    assert urdf.read_text()==original
    assert ET.parse(output).find('.//mesh').get('filename')==str(mesh.resolve())
    with pytest.raises(ValueError,match='new'):
        resolve_meshes(urdf,output,[str(root)])


def test_unresolved_cad_fails_before_partial_output(tmp_path):
    urdf=tmp_path/'original.urdf'
    urdf.write_text('<robot><link><visual><geometry><mesh filename="package://missing/shape.stl"/></geometry></visual></link></robot>')
    with pytest.raises(ValueError,match='unresolved'):
        resolve_meshes(urdf,tmp_path/'out.urdf',[])
    assert not (tmp_path/'out.urdf').exists()
