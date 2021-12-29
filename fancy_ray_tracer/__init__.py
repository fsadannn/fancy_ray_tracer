from .camera import Camera
from .canvas import Canvas
from .illumination import Light, lighting, reflect
from .materials import (
    ChessPattern,
    LinearGradient,
    Material,
    RingPatter,
    StripePattern,
    make_material,
)
from .matrices import rotX, rotY, rotZ, scaling, sharing, translation, view_transform
from .primitives import Cube, Plane, Sphere, glass_sphere
from .protocols import WorldObject
from .ray import Computations, Intersection, Ray, hit, hit_sorted, normal_at
from .tuples import make_color, normalize, point, vector
from .utils import chain, chain_ops, equal
from .world import World, schlick

__version__ = '0.0.1'
