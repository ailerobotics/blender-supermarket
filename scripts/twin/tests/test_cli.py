"""Argument handling + dry-run rendering."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from twin import cli  # noqa: E402


def test_defaults():
    args = cli.build_parser().parse_args([])
    assert args.store.endswith("export/supermarket.usdc")
    assert args.robot_urdf.endswith("shelfbot_gripper_isaac.urdf")
    assert args.spawn_aisle == "Cam_Aisle"
    assert args.headless is True
    assert args.product_collider == "convex"
    assert args.dry_run is False
    assert args.vram_budget_mb == 8000


def test_no_headless_and_force_and_spawn_xy():
    args = cli.build_parser().parse_args(
        ["--no-headless", "--force", "--spawn-xy", "-1.52", "0.75", "--product-collider", "none"])
    assert args.headless is False
    assert args.force is True
    assert args.spawn_xy == [-1.52, 0.75]
    assert args.product_collider == "none"


def test_screenshot_defaults_from_save():
    args = cli.build_parser().parse_args(["--save", "/tmp/foo.usd"])
    assert cli.resolve_screenshot(args) == "/tmp/foo.png"


def test_invalid_product_collider_rejected():
    import pytest
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--product-collider", "rigid"])


def _plan():
    return {
        "store": "/s/supermarket.usdc", "robot_urdf": "/r/robot.urdf", "robot_usd": None,
        "usd_facts": {"up_axis": "Z", "meters_per_unit": 1.0, "default_prim": "/Supermarket",
                      "floor_z": 0.0,
                      "collider_targets": {"triangle_mesh_meshes": 702, "structure": 10,
                                           "fixtures": 674, "signage": 18, "prototypes": 196,
                                           "instances": 24927}},
        "spawn": {"label": "Cam_Aisle aisle", "x": -1.52, "y": 0.75, "z": 0.0,
                  "clearance_m": 0.9, "root_translate": (-1.52, 1.95, 0.0)},
        "collider_plan": {"triangle_mesh": 702, "convex_hull": 197, "skipped_instances": 24927},
        "product_collider": "convex", "min_clearance": 0.6, "yaw_deg": 0.0,
        "save": "/tmp/store_with_robot.usd", "screenshot": "/tmp/store_with_robot.png",
        "gpu": {"ok": False, "message": "REFUSING to boot Isaac Sim: ... blender ..."},
    }


def test_render_plan_mentions_no_boot_and_key_numbers():
    text = cli.render_plan(_plan())
    assert "DRY RUN" in text
    assert "no GPU session was started" in text
    assert "URDF -> USD" in text and "CANNOT run without a simulator boot" in text
    assert "702" in text and "196" in text and "24927" in text
    assert "-1.52" in text and "0.9" in text
    assert "REFUSING" in text


def test_render_plan_flags_nonzero_floor():
    p = _plan()
    p["usd_facts"]["floor_z"] = 0.4
    text = cli.render_plan(p)
    assert "NOT 0" in text


def test_render_plan_preconverted_robot_skips_import():
    p = _plan()
    p["robot_usd"] = "/r/robot.usd"
    text = cli.render_plan(p)
    assert "import step SKIPPED" in text
