"""Single source of truth for WHERE a mission-config key is read from.

sim-preflight-nested-cart-placement-v1.

The Isaac manipulation factory (``isaac_factory.parse_manipulation_config``) reads the
collect cart placement from the NESTED block ``config['manipulation']['cart_placement']``
of the mission RUNTIME config (``mission-runtime-*.json``); a TOP-LEVEL
``config['cart_placement']`` is IGNORED. coldwin42-47 edited the top-level block, so
``support_prim`` / ``support_z_world`` / ``load_center_body_xy`` never reached the factory
(coldwin47 logged ``PLACE_SUPPORT_SELECT selector=None support_z 0.9325`` and placed on the
deck though the shelf was configured) and FIVE GPU runs were spent on settings the
simulation never read. Worse, ``config_preflight`` validated the top-level block, so the
gate passed on the block the factory ignores.

The fleet rule is that a key in the wrong file OR the wrong block is silently inactive, and
``config_preflight`` is the gate that must catch it. So both the factory and the preflight
resolve the cart placement through the ONE accessor here -- they can never again disagree on
the source of a key.

Pure / CPU: stdlib only, no Isaac, no usd-core.
"""
from __future__ import annotations

#: the nested RUNTIME block the manipulation factory parses.
MANIPULATION_BLOCK = 'manipulation'
#: the placement key read from inside that block.
CART_PLACEMENT_KEY = 'cart_placement'
#: dotted path of the cart_placement block the factory actually reads (for logs/messages).
CART_PLACEMENT_SOURCE = f'{MANIPULATION_BLOCK}.{CART_PLACEMENT_KEY}'


def manipulation_block(config):
    """The nested manipulation block (``config['manipulation']``) as a dict, or ``{}``."""
    block = (config or {}).get(MANIPULATION_BLOCK)
    return block if isinstance(block, dict) else {}


def cart_placement_block(config):
    """The cart placement the Isaac factory reads: ``config['manipulation']['cart_placement']``.

    Returns the value as authored (a dict / ``None`` / whatever is present); parse-time
    validation stays in ``isaac_factory._parse_cart_placement``. The TOP-LEVEL
    ``config['cart_placement']`` is NOT read here -- it is the ignored block (coldwin42-47)."""
    return manipulation_block(config).get(CART_PLACEMENT_KEY)


def top_level_cart_placement(config):
    """The IGNORED top-level ``config['cart_placement']`` (what coldwin42-47 edited by
    mistake), or ``None`` when absent. Only ``config_preflight`` reads this -- to refuse a
    stale duplicate -- never the factory."""
    return (config or {}).get(CART_PLACEMENT_KEY)
