from typing import Dict, List, TextIO, Tuple

import numpy as np

from fancy_ray_tracer.primitives import Group, Triangle, TriangleMesh

from .protocols import TriangleFaces
from .tuples import point, vector


def fan_triangulation(vertices: List[int]) -> TriangleFaces:
    triangles: TriangleFaces = []
    v0 = vertices[0]
    for i in range(1, len(vertices) - 1):
        triangles.append((v0, vertices[i], vertices[i + 1]))

    return triangles


class WavefrontOBJ:
    def __init__(self, data: TextIO) -> None:
        self.data = data

    def parse(self) -> Group:
        vertices: List[np.ndarray] = []
        normals: List[np.ndarray] = []
        # textures: List[np.ndarray] = []

        faces: TriangleFaces = []
        faces_normals: TriangleFaces = []
        # TODO: uncomment when implement textures
        # faces_textures: TriangleFaces = []

        current_group = ""
        faces_groups: Dict[str, TriangleFaces] = {}
        normals_groups: Dict[str, TriangleFaces] = {}
        # TODO: uncomment when implement textures
        # texture_groups: Dict[str, TriangleFaces] = {}

        for line in self.data:
            line = tuple(filter(lambda x: x != '', line.strip().split(' ')))
            if len(line) == 0 or line[0] == '#':
                continue
            if len(line[0]) == 1:
                if len(line) == 4 and line[0] == 'v':
                    p = point(float(line[1]), float(line[2]), float(line[3]))
                    vertices.append(p)
                if line[0] == 'f':
                    other_format = '/' in line[1]
                    if len(line) == 4 and not other_format:
                        faces.append(
                            (int(line[1]) - 1, int(line[2]) - 1, int(line[3]) - 1))
                    else:
                        if other_format:
                            fc = []
                            nm = []
                            # TODO: uncomment when implement textures
                            # tx = []
                            for i in line[1:]:
                                vals = i.split('/')
                                fc.append(int(vals[0]) - 1)
                                nm.append(int(vals[2]) - 1)
                                # TODO: uncomment when implement textures
                                # if vals[1]!='':
                                #     tx.append(int(vals[1])-1)
                            if len(fc) == 3:
                                faces.append(tuple(fc))
                                faces_normals.append(tuple(nm))
                                # TODO: uncomment when implement textures
                                # if len(tx)!=0:
                                #     textures.append(tuple)
                            else:
                                res = fan_triangulation(fc)
                                faces.extend(res)
                                res = fan_triangulation(nm)
                                faces_normals.extend(res)
                                # TODO: uncomment when implement textures
                                # res = fan_triangulation(textures)
                                # faces_textures.extend(res)

                        else:
                            vc = tuple(int(i) for i in line[1:])
                            faces.extend(fan_triangulation(vc))

                if line[0] == 'g':
                    if len(faces) != 0:
                        faces_groups[current_group] = faces
                        faces = []
                    if len(faces_normals) != 0:
                        normals_groups[current_group] = faces_normals
                        faces_normals = []
                    # TODO: uncomment when implement textures
                    # if len(textures) != 0:
                    #     texture_groups[current_group] = textures
                    #     textures = []
                    current_group = line[1]
            elif len(line[0]) == 2:
                if line[0] == 'vn':
                    vn = vector(float(line[1]), float(line[2]), float(line[3]))
                    normals.append(vn)
                # TODO: uncomment when implement textures
                # if line[0] == 'vt':
                #     vn = vector(float(line[1]), float(line[2]), float(line[3]))
                #     textures.append(vn)
                pass

        if len(faces) != 0:
            faces_groups[current_group] = faces
        if len(faces_normals) != 0:
            normals_groups[current_group] = faces_normals
        # TODO: uncomment when implement textures
        # if len(textures) != 0:
        #     texture_groups[current_group] = textures

        g = Group()
        if len(faces_groups) == 1:
            if len(normals) != 0:
                mesh = TriangleMesh(vertices, faces,
                                    normals, normals_groups[current_group])
                g.add_shape(mesh)
            else:
                for face in faces:
                    p1 = vertices[face[0]]
                    p2 = vertices[face[1]]
                    p3 = vertices[face[2]]
                    t = Triangle(p1, p2, p3)
                    g.add_shape(t)
            return g

        for gname, faces in faces_groups.items():
            if gname in normals_groups:
                mesh = TriangleMesh(vertices, faces,
                                    normals, normals_groups[gname])
                g.add_shape(mesh)
            else:
                gm = Group()
                for face in faces:
                    p1 = vertices[face[0]]
                    p2 = vertices[face[1]]
                    p3 = vertices[face[2]]
                    t = Triangle(p1, p2, p3)
                    gm.add_shape(t)
                g.add_shape(gm)
        return g
