import numpy as np
from chroma.itertoolset import *
from chroma.geometry import Mesh, Solid, Geometry
from chroma.make import sphere

from chroma.demo.optics import *

def build_checkerboard_scene(checkers_per_side=10, squares_per_checker=50):
    x = np.linspace(-5000.0, 5000.0, checkers_per_side*squares_per_checker+1)
    y = np.linspace(-5000.0, 5000.0, checkers_per_side*squares_per_checker+1)

    vertices = np.array(tuple(product(x,y,[0])))

    triangles = []
    for j in range(y.size-1):
        for i in range(x.size-1):
            triangles.append([j*len(x)+i, (j+1)*len(x)+i,(j+1)*len(x)+i+1]) 
            triangles.append([j*len(x)+i, j*len(x)+i+1,(j+1)*len(x)+i+1]) 

    checkerboard_mesh = Mesh(vertices, triangles, remove_duplicate_vertices=True)

    checkerboard_color_line1 = take(checkers_per_side*squares_per_checker*2, cycle([0]*2*squares_per_checker + [0xffffff]*2*squares_per_checker))*squares_per_checker
    checkerboard_color_line2 = take(checkers_per_side*squares_per_checker*2, cycle([0xffffff]*2*squares_per_checker + [0]*2*squares_per_checker))*squares_per_checker
    checkerboard_color = take(len(checkerboard_mesh.triangles), cycle(checkerboard_color_line1 + checkerboard_color_line2))

    checkerboard_surface_line1 = take(checkers_per_side*squares_per_checker*2, cycle([black_surface]*2*squares_per_checker + [lambertian_surface]*2*squares_per_checker))*squares_per_checker
    checkerboard_surface_line2 = take(checkers_per_side*squares_per_checker*2, cycle([lambertian_surface]*2*squares_per_checker + [black_surface]*2*squares_per_checker))*squares_per_checker
    checkerboard_surface = take(len(checkerboard_mesh.triangles), cycle(checkerboard_surface_line1 + checkerboard_surface_line2))

    checkerboard = Solid(checkerboard_mesh, vacuum, vacuum, surface=checkerboard_surface, color=checkerboard_color)

    sphere1 = Solid(sphere(1000.0, nsteps=512), water, vacuum)
    sphere2 = Solid(sphere(1000.0, nsteps=512), vacuum, vacuum, 
                    surface=shiny_surface)
    sphere3 = Solid(sphere(1000.0, nsteps=512), vacuum, vacuum, surface=lambertian_surface)

    checkerboard_scene = Geometry()
    checkerboard_scene.add_solid(checkerboard, displacement=(0,0,-1500.0))
    checkerboard_scene.add_solid(sphere1, displacement=(2000.0,-2000.0,0))
    checkerboard_scene.add_solid(sphere2, displacement=(-2000.0,-2000.0,0))
    checkerboard_scene.add_solid(sphere3, displacement=(0.0,2000.0,0))

    return checkerboard_scene
