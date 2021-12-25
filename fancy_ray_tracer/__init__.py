from .canvas import Canvas, CanvasImg
from .materials import Material, make_material
from .matrices import rotX, rotY, rotZ, scaling, sharing, translation
from .primitives import Sphere, make_sphere
from .protocols import WorldObject
from .ray import (
    Intersection,
    Light,
    Ray,
    hit,
    intersect,
    lighting,
    normal_at,
    reflect,
    transform,
)
from .tuples import make_color, normalize, point, vector
from .utils import chain, chain_ops, equal

__version__ = '0.0.1'
