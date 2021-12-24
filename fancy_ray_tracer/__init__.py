from .canvas import Canvas, CanvasImg
from .matrices import rotX, rotY, rotZ, scaling, sharing, translation
from .ray import (
    Intersection,
    Light,
    Material,
    Ray,
    Sphere,
    hit,
    intersect,
    lighting,
    make_material,
    make_sphere,
    normal_at,
    reflect,
    transform,
)
from .tuples import equal, point, vector
from .utils import chain, chain_ops, rand_id

__version__ = '0.0.1'
