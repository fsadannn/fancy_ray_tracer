from .canvas import Canvas, CanvasImg
from .ilumination import Light, lighting, reflect
from .materials import Material, make_material
from .matrices import rotX, rotY, rotZ, scaling, sharing, translation
from .primitives import Sphere, make_sphere
from .protocols import WorldObject
from .ray import Intersection, Ray, hit, normal_at
from .tuples import make_color, normalize, point, vector
from .utils import chain, chain_ops, equal
from .world import World

__version__ = '0.0.1'
