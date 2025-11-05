"""
2D L-bracket mesh generator using the Gmsh Python API.
Produces a triangular FEM mesh suitable for plane stress/strain analysis.
Outputs: bracket_2d.msh and bracket_2d.vtk
"""

import gmsh
import math
import sys


def make_2d_bracket(out_msh="bracket_2d.msh",
                    leg1=60.0,    # horizontal leg length
                    leg2=40.0,    # vertical leg length
                    thickness=5.0,# bracket arm thickness
                    hole_r=5.0,   # optional circular hole near corner
                    mesh_size=2.5 # characteristic mesh element size
                    ):
    gmsh.initialize()
    gmsh.model.add("L_bracket_2D")

    # Helper: add point with characteristic length
    def P(x, y, lc=mesh_size):
        return gmsh.model.geo.addPoint(x, y, 0, lc)

    # Define corner points of L-bracket
    p0 = P(0, 0)                     # inner corner
    p1 = P(leg1, 0)                  # right end
    p2 = P(leg1, thickness)          # up a bit
    p3 = P(thickness, thickness)     # inner notch
    p4 = P(thickness, leg2)          # top of vertical leg
    p5 = P(0, leg2)                  # left end

    # Create boundary lines
    l = []
    l.append(gmsh.model.geo.addLine(p0, p1))
    l.append(gmsh.model.geo.addLine(p1, p2))
    l.append(gmsh.model.geo.addLine(p2, p3))
    l.append(gmsh.model.geo.addLine(p3, p4))
    l.append(gmsh.model.geo.addLine(p4, p5))
    l.append(gmsh.model.geo.addLine(p5, p0))

    outer_loop = gmsh.model.geo.addCurveLoop(l)
    surface_loops = [outer_loop]

    # Optional circular hole near the inner corner
    if hole_r > 0:
        # Center of hole slightly offset from the inner corner
        cx = thickness + hole_r + 2.0
        cy = thickness + hole_r + 2.0

        # Define hole points (3 points define a full circle)
        ph1 = P(cx + hole_r, cy)
        ph2 = P(cx, cy + hole_r)
        ph3 = P(cx - hole_r, cy)
        ph4 = P(cx, cy - hole_r)
        center = P(cx, cy)

        # Four circle arcs to make the hole loop
        a1 = gmsh.model.geo.addCircleArc(ph1, center, ph2)
        a2 = gmsh.model.geo.addCircleArc(ph2, center, ph3)
        a3 = gmsh.model.geo.addCircleArc(ph3, center, ph4)
        a4 = gmsh.model.geo.addCircleArc(ph4, center, ph1)

        hole_loop = gmsh.model.geo.addCurveLoop([a1, a2, a3, a4])
        surface_loops.append(hole_loop)

    # Define the final surface (outer boundary with optional inner hole)
    surf = gmsh.model.geo.addPlaneSurface(surface_loops)

    # Synchronize geometry
    gmsh.model.geo.synchronize()

    # --- Physical groups (useful for FEM BCs) ---
    gmsh.model.addPhysicalGroup(2, [surf], tag=1)
    gmsh.model.setPhysicalName(2, 1, "L_bracket_2D")

    gmsh.model.addPhysicalGroup(1, [l[0]], tag=11)
    gmsh.model.setPhysicalName(1, 11, "bottom_edge")

    gmsh.model.addPhysicalGroup(1, [l[5]], tag=12)
    gmsh.model.setPhysicalName(1, 12, "left_edge")

    gmsh.model.addPhysicalGroup(1, [l[1], l[2], l[3], l[4]], tag=13)
    gmsh.model.setPhysicalName(1, 13, "outer_edges")

    # --- Mesh options ---
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 1.5)

    # Generate 2D mesh
    gmsh.model.mesh.generate(2)

    # Write output files
    gmsh.write(out_msh)
    print(f"Wrote mesh to {out_msh}")

    try:
        gmsh.write("bracket_2d.vtk")
        print("Also wrote bracket_2d.vtk")
    except Exception:
        pass

    gmsh.finalize()


if __name__ == "__main__":
    make_2d_bracket()
