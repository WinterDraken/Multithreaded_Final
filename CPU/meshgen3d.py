"""
3D L-bracket mesh generator using Gmsh Python API.
Extrudes a 2D L-shaped profile into a solid and meshes it with tetrahedra.
"""

import gmsh


def make_3d_bracket(out_msh="bracket_3d.msh",
                    leg1=60.0,
                    leg2=40.0,
                    web=5.0,
                    mesh_size=3.0,
                    extrude_thickness=10.0):
    gmsh.initialize()
    gmsh.model.add("L_bracket_3D")

    #Define 2D profile (L shape)
    def P(x, y, lc=mesh_size):
        return gmsh.model.geo.addPoint(x, y, 0, lc)

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

    #Extrude 2D surface into 3D volume
    dx, dy, dz = 0, 0, extrude_thickness
    extruded = gmsh.model.geo.extrude([(2, surf)], dx, dy, dz)

    vol = None
    for ent in extruded:
        if ent[0] == 3:
            vol = ent[1]
            break
    if vol is None:
        raise RuntimeError("Extrusion failed: no volume found.")

    gmsh.model.geo.synchronize()

    #Physical groups
    gmsh.model.addPhysicalGroup(3, [vol], tag=1)
    gmsh.model.setPhysicalName(3, 1, "L_bracket_volume")

    # Identify top and bottom faces by z-extents
    boundary_faces = gmsh.model.getBoundary([(3, vol)], oriented=False, recursive=False)
    bottom_faces, top_faces = [], []
    for dim, tag in boundary_faces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        zcenter = 0.5 * (zmin + zmax)
        if zcenter < 1e-6:
            bottom_faces.append(tag)
        elif abs(zcenter - extrude_thickness) < 1e-6:
            top_faces.append(tag)

    if bottom_faces:
        gmsh.model.addPhysicalGroup(2, bottom_faces, tag=11)
        gmsh.model.setPhysicalName(2, 11, "bottom_face")
    if top_faces:
        gmsh.model.addPhysicalGroup(2, top_faces, tag=12)
        gmsh.model.setPhysicalName(2, 12, "top_face")

    # Mesh options
    gmsh.option.setNumber("Mesh.Algorithm3D", 4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 1.2)

    #Generate mesh
    gmsh.model.mesh.generate(3)
    gmsh.write(out_msh)
    print(f"Wrote {out_msh}")

    try:
        gmsh.write("bracket_3d.vtk")
        print("Also wrote bracket_3d.vtk")
    except Exception:
        pass

    gmsh.finalize()


if __name__ == "__main__":
    make_3d_bracket()
