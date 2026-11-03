"""Make the host's move_group robot_description mesh references loadable (CPU, no GPU).

The host stack writes move_group's ``robot_description`` from the generated
``mobile.urdf`` (``restock_host.generate_assets`` resolves every ``package://`` mesh to a
bare ABSOLUTE filesystem path, e.g.
``/opt/ros/jazzy/share/ur_description/meshes/ur10/collision/base.stl``). MoveIt loads every
link visual/collision mesh through ``resource_retriever``, which treats a ``<mesh filename>``
as a URI:

  * ``package://pkg/rel`` -> resolved against the ament index (loads);
  * ``file:///abs/path``  -> opened directly (loads);
  * a bare path with NO scheme -> handed to the URL layer, which REJECTS it. move_group's
    C++/libcurl retriever logs ``URL rejected: No host part in the URL`` for every such mesh;
    the Python ``resource_retriever`` raises ``ValueError: unknown url type`` -- same root
    cause (no URI scheme), confirmed against the real retriever.

Net effect on the resolved URDF the host ships today: move_group loads NO arm-link collision
or visual geometry. The cuMotion XRDF spheres (read from ``urdf_file_path``, a separate
loader) still do the real self/scene collision checking, and MoveIt's FCL ``ValidateSolution``
runs against EMPTY link geometry. Rewriting each bare absolute mesh path to a ``file://`` URI
gives the retriever a loadable scheme without moving any file or changing any geometry -- the
pointed-at ``.stl``/``.dae`` bytes are identical.

This module is a pure CPU helper: the one URI rewrite ``normalize_mesh_uris`` the host applies
to the robot_description string (behind the opt-in ``moveit_mesh_uris`` HOST config key), and
a faithful model of the retriever's URI classification (``retriever_loadable`` /
``loadable_meshes``) so the fix is provable without a ROS/GPU boot.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Callable, List, Tuple


def retriever_loadable(filename: str, *,
                       exists: Callable[[str], bool] = os.path.isfile) -> Tuple[bool, str]:
    """Model how MoveIt's ``resource_retriever`` classifies one mesh ``filename``.

    Returns ``(loadable, reason)``. ``reason`` is empty when loadable. This mirrors the real
    retriever (verified against ``resource_retriever.get`` on ur_description meshes):

      * ``package://pkg/rel`` -> loadable (ament-resolved; we do not re-resolve the index here,
        the generated URDF never ships package:// anyway -- generate_assets resolves them);
      * ``file://...``        -> loadable when the pointed-at file exists, else a missing-file
        reason (NOT the no-scheme rejection this fix targets);
      * a bare path (absolute or relative) -> NOT loadable: the retriever rejects a filename
        with no URI scheme ("URL rejected: No host part in the URL" in move_group's log).
    """
    if not filename:
        return False, 'empty mesh filename'
    if filename.startswith('package://'):
        return True, ''
    if filename.startswith('file://'):
        path = filename[len('file://'):]
        if exists(path):
            return True, ''
        return False, 'file:// URI points at a missing file: ' + path
    # No URI scheme: this is exactly the defect. The C++/libcurl retriever move_group uses
    # logs "URL rejected: No host part in the URL"; the Python retriever raises "unknown url
    # type". Either way the mesh never loads.
    return False, 'URL rejected: No host part in the URL (bare path, no URI scheme): ' + filename


def _mesh_filenames(urdf_text: str, role: str | None = None) -> List[str]:
    root = ET.fromstring(urdf_text)
    out: List[str] = []
    for link in root.findall('link'):
        for tag in (('visual', 'collision') if role is None else (role,)):
            for element in link.findall(tag):
                for mesh in element.findall('.//mesh'):
                    name = mesh.get('filename')
                    if name is not None:
                        out.append(name)
    return out


def loadable_meshes(urdf_text: str, *, role: str | None = None,
                    exists: Callable[[str], bool] = os.path.isfile) -> List[str]:
    """Distinct mesh filenames in ``urdf_text`` that the retriever model would LOAD.

    ``role`` restricts to ``'collision'`` or ``'visual'`` (default: both). This is the
    loader-level predicate behind the fix: empty for the bare-path URDF the host ships today,
    non-empty once the filenames carry a ``file://`` scheme.
    """
    return [f for f in _mesh_filenames(urdf_text, role) if retriever_loadable(f, exists=exists)[0]]


def normalize_mesh_uris(urdf_text: str) -> str:
    """Return ``urdf_text`` with every bare ABSOLUTE ``<mesh filename>`` rewritten to a
    ``file://`` URI; ``package://`` and existing ``file://`` references are left untouched.

    The rewrite is textual only -- no geometry, link, joint, inertial or transform is changed,
    and no mesh file is read or moved. A relative bare path is left as-is (the host never ships
    one; rewriting it would invent a working directory and could hide a genuinely unresolved
    mesh). Idempotent: re-applying it is a no-op.
    """
    root = ET.fromstring(urdf_text)
    for mesh in root.findall('.//mesh'):
        name = mesh.get('filename')
        if name is None:
            continue
        if name.startswith('package://') or name.startswith('file://'):
            continue
        if os.path.isabs(name):
            mesh.set('filename', 'file://' + name)
    return ET.tostring(root, encoding='unicode')
