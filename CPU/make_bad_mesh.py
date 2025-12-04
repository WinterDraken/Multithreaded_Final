#!/usr/bin/env python3
"""
Make a 'bad' version of a Gmsh mesh by randomly renumbering all node tags.

Usage:
    python make_bad_mesh.py bracket_3d_large.msh
    # -> writes bracket_3d_large_bad.msh

    python make_bad_mesh.py bracket_3d_large.msh custom_name.msh
    # -> writes custom_name.msh
"""

import sys
import os
import random
import gmsh


def make_bad_mesh(in_msh, out_msh=None, seed=0):
    if out_msh is None:
        root, ext = os.path.splitext(in_msh)
        out_msh = root + "_bad" + ext

    gmsh.initialize()
    try:
        print(f"[INFO] Loading mesh: {in_msh}")
        gmsh.open(in_msh)

        # Get all node tags in the mesh
        nodeTags, coords, _ = gmsh.model.mesh.getNodes()
        old = list(nodeTags)

        # Build a random permutation of node tags
        new = old.copy()
        random.seed(seed)     # use fixed seed for reproducibility; remove if you want different each time
        random.shuffle(new)

        print(f"[INFO] Renumbering {len(old)} nodes with a random permutation...")
        gmsh.model.mesh.renumberNodes(old, new)
        # Rebuild caches so everything stays consistent
        gmsh.model.mesh.rebuildNodeCache()
        gmsh.model.mesh.rebuildElementCache()

        print(f"[INFO] Writing bad mesh to: {out_msh}")
        gmsh.write(out_msh)
    finally:
        gmsh.finalize()

    print("[DONE] Bad mesh created.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python make_bad_mesh.py input.msh [output.msh]")
        sys.exit(1)

    in_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) >= 3 else None
    make_bad_mesh(in_file, out_file)
