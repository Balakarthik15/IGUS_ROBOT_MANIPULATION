#!/usr/bin/env python3
"""Generate Gazebo models of Duplo-size building blocks for the thesis setup.

Writes one model folder per block type and colour:
    <out>/block_2x4_red/model.config
    <out>/block_2x4_red/model.sdf
    ...

Sizes made by default (your pick area):
    2x4 with studs     (target blocks)
    2x4 without studs  (distractor)
    2x2 with studs     (distractor)
    2x6 with studs     (distractor)
in red, green, yellow and blue.

The model origin is the CENTRE OF THE BOTTOM FACE, so a block placed at the
table height z sits exactly on the table.

Run:  python3 make_blocks.py ~/IGUS_ROBOT_MANIPULATION/src/<your_pkg>/models
"""
import os
import sys

# ---- dimensions (metres). Measured block 63.8 x 31.8 x 19 mm. ----
PITCH = 0.01595          # one stud step, 4 steps = 63.8 mm, 2 steps = 31.9 mm
BODY_H = 0.019           # body height without studs
STUD_D = 0.0094          # stud diameter (approximate, measure a real block)
STUD_H = 0.0045          # stud height (approximate, measure a real block)
MASS_2X4 = 0.012         # kg, approximate, weigh a real block
FRICTION = 1.0

COLORS = {
    "red":    "0.80 0.08 0.08 1",
    "green":  "0.10 0.60 0.15 1",
    "yellow": "0.95 0.80 0.05 1",
    "blue":   "0.08 0.30 0.80 1",
}

# (studs along x, studs along y, has studs)
BLOCK_TYPES = [
    (4, 2, True),
    (4, 2, False),
    (2, 2, True),
    (6, 2, True),
]


def model_name(nx, ny, studs, color):
    size = f"{ny}x{nx}"
    return f"block_{size}{'' if studs else '_nostud'}_{color}"


def make_sdf(name, nx, ny, studs, rgba):
    lx, ly, lz = nx * PITCH, ny * PITCH, BODY_H
    mass = MASS_2X4 * (nx * ny) / 8.0
    ixx = mass / 12.0 * (ly ** 2 + lz ** 2)
    iyy = mass / 12.0 * (lx ** 2 + lz ** 2)
    izz = mass / 12.0 * (lx ** 2 + ly ** 2)

    material = f"""<material>
            <ambient>{rgba}</ambient>
            <diffuse>{rgba}</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>"""
    surface = f"""<surface>
            <friction><ode><mu>{FRICTION}</mu><mu2>{FRICTION}</mu2></ode></friction>
          </surface>"""

    parts = []
    parts.append(f"""
        <collision name="body_collision">
          <pose>0 0 {lz/2:.5f} 0 0 0</pose>
          <geometry><box><size>{lx:.5f} {ly:.5f} {lz:.5f}</size></box></geometry>
          {surface}
        </collision>
        <visual name="body_visual">
          <pose>0 0 {lz/2:.5f} 0 0 0</pose>
          <geometry><box><size>{lx:.5f} {ly:.5f} {lz:.5f}</size></box></geometry>
          {material}
        </visual>""")

    if studs:
        k = 0
        for i in range(nx):
            for j in range(ny):
                x = (i - (nx - 1) / 2.0) * PITCH
                y = (j - (ny - 1) / 2.0) * PITCH
                z = lz + STUD_H / 2.0
                parts.append(f"""
        <collision name="stud{k}_collision">
          <pose>{x:.5f} {y:.5f} {z:.5f} 0 0 0</pose>
          <geometry><cylinder><radius>{STUD_D/2:.5f}</radius><length>{STUD_H:.5f}</length></cylinder></geometry>
        </collision>
        <visual name="stud{k}_visual">
          <pose>{x:.5f} {y:.5f} {z:.5f} 0 0 0</pose>
          <geometry><cylinder><radius>{STUD_D/2:.5f}</radius><length>{STUD_H:.5f}</length></cylinder></geometry>
          {material}
        </visual>""")
                k += 1

    return f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{name}">
    <link name="link">
      <inertial>
        <pose>0 0 {lz/2:.5f} 0 0 0</pose>
        <mass>{mass:.5f}</mass>
        <inertia>
          <ixx>{ixx:.4e}</ixx><iyy>{iyy:.4e}</iyy><izz>{izz:.4e}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>{''.join(parts)}
    </link>
  </model>
</sdf>
"""


def make_config(name):
    return f"""<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.8">model.sdf</sdf>
  <description>Duplo-size building block for the igus ReBeL pick-and-place simulation.</description>
</model>
"""


def main():
    out = os.path.abspath(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "models"))
    made = []
    for nx, ny, studs in BLOCK_TYPES:
        for color, rgba in COLORS.items():
            name = model_name(nx, ny, studs, color)
            d = os.path.join(out, name)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "model.sdf"), "w") as f:
                f.write(make_sdf(name, nx, ny, studs, rgba))
            with open(os.path.join(d, "model.config"), "w") as f:
                f.write(make_config(name))
            made.append(name)
    print(f"wrote {len(made)} models to {out}")
    for n in made:
        print("  ", n)


if __name__ == "__main__":
    main()