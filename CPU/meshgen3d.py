"""
Large 3D L-bracket mesh generator using Gmsh Python API.
Scales up the 2D L-shaped profile and extrudes it into a solid mesh.
"""

import gmsh


def make_3d_bracket(out_msh="bracket_3d_large.msh",
                    leg1=200.0,        # horizontal leg (mm)
                    leg2=200.0,        # vertical leg   (mm)
                    web=12.0,          # thickness of the L rib
                    mesh_size=4.0,     # coarse mesh (2.0 for fine)
                    extrude_thickness=50.0):  # bracket depth
    gmsh.initialize()
    gmsh.model.add("L_bracket_3D_large")

    # --- Define 2D L-shape profile --- 
    def P(x, y, lc=mesh_size):
        return gmsh.model.geo.addPoint(x, y, 0, lc)

    # Larger geometry
    p0 = P(0, 0)
    p1 = P(leg1, 0)
    p2 = P(leg1, web)
    p3 = P(web, web)
    p4 = P(web, leg2)
    p5 = P(0, leg2)

    lines = [
        gmsh.model.geo.addLine(p0, p1),
        gmsh.model.geo.addLine(p1, p2),
        gmsh.model.geo.addLine(p2, p3),
        gmsh.model.geo.addLine(p3, p4),
        gmsh.model.geo.addLine(p4, p5),
        gmsh.model.geo.addLine(p5, p0)
    ]

    loop = gmsh.model.geo.addCurveLoop(lines)
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    # --- Extrude to 3D ---
    dx = dy = 0
    dz = extrude_thickness
    extruded = gmsh.model.geo.extrude([(2, surf)], dx, dy, dz)

    volume = None
    for ent in extruded:
        if ent[0] == 3:
            volume = ent[1]
            break
    if volume is None:
        raise RuntimeError("Extrusion failed!")

    gmsh.model.geo.synchronize()

    # --- Physical Groups ---
    gmsh.model.addPhysicalGroup(3, [volume], tag=1)
    gmsh.model.setPhysicalName(3, 1, "Large_L_bracket_volume")

    # Detect top and bottom faces
    faces = gmsh.model.getBoundary([(3, volume)], oriented=False, recursive=False)
    bottom_faces, top_faces = [], []
    for dim, tag in faces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        if abs(zmin) < 1e-6:
            bottom_faces.append(tag)
        elif abs(zmax - extrude_thickness) < 1e-6:
            top_faces.append(tag)

    if bottom_faces:
        gmsh.model.addPhysicalGroup(2, bottom_faces, tag=11)
        gmsh.model.setPhysicalName(2, 11, "bottom_face")
    if top_faces:
        gmsh.model.addPhysicalGroup(2, top_faces, tag=12)
        gmsh.model.setPhysicalName(2, 12, "top_face")

    # --- Mesh Settings ---
    gmsh.option.setNumber("Mesh.Algorithm3D", 4)  # Delaunay
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.7)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 1.3)

    # --- Generate ---
    gmsh.model.mesh.generate(3)
    gmsh.write(out_msh)
    print(f"Mesh written: {out_msh}")

    try:
        gmsh.write(out_msh.replace(".msh", ".vtk"))
        print("VTK exported.")
    except:
        pass

    gmsh.finalize()


if __name__ == "__main__":
    make_3d_bracket()
