#!/usr/bin/env python3
"""Generate Gazebo models of the lab's Duplo-size blocks (version 2, lab table of Oct 2026).

Main blocks (the 5 YOLO classes): studded 2x4 in red, green, yellow, white, lightblue.

Distractors (27, all stand studs up):
  lab pieces (14, the ones that exist in the lab):
    2x2 red, green, yellow, white, skyblue, black
    2x6 white, blue
    2x4 lime, blue, orange
    4x4 yellow
    step_2x4_lime        lime 2x4 with a lower step at both ends
    step_2x3_paleyellow  pale yellow 2x3 with a lower step at one end
  sim-only extras (13):
    nostud_2x4 in the 5 main colours (2x4 without studs)
    2x2 lightblue, 2x6 red, green, yellow, lightblue
    2x4 paleyellow, skyblue, cream (near colours)

Also writes blocks_manifest.json with every model's name, type, size and colour,
so the world file and the data generator use the same list.

The model origin is the CENTRE OF THE BOTTOM, so z = surface height puts a block on it.
The long side of a block is along the model's x axis.

Run from the package folder:  python3 make_blocks.py models
"""
import json
import os
import sys

# ---- Duplo dimensions (metres) ----
PITCH = 0.01595          # one stud step: 4 steps = 63.8 mm, 2 steps = 31.9 mm
BODY_H = 0.019           # body height without studs
STEP_H = 0.0095          # height of the lower step on the step pieces
STUD_D = 0.0094          # stud outer diameter
STUD_H = 0.0045          # stud height
RING = 0.62              # inner (darker) disc of a hollow stud, as a share of the stud diameter
WALL = 0.0015            # wall thickness of an upside-down block
MASS_2X4 = 0.012         # kg, scaled by footprint for other sizes
FRICTION = 1.0

# ---- colours (0..1). Main colours measured from the lab camera image. ----
COLORS = {
    'red':        (0.82, 0.02, 0.13),
    'green':      (0.29, 0.67, 0.45),
    'yellow':     (0.99, 0.77, 0.29),
    'white':      (0.97, 0.97, 0.95),   # camera shows 1.00 (overexposed), kept slightly below
    'lightblue':  (0.78, 0.93, 0.93),   # camera shows 0.88 1.00 1.00 (overexposed), kept apart from white
    # distractor-only colours, from the phone photo (approximate)
    'lime':       (0.67, 0.86, 0.26),
    'paleyellow': (0.96, 0.93, 0.62),
    'skyblue':    (0.15, 0.75, 0.92),
    'blue':       (0.05, 0.49, 0.79),
    'orange':     (0.98, 0.36, 0.14),   # from the lab camera images, brightened like the others
    'black':      (0.10, 0.09, 0.11),
    'cream':      (0.93, 0.90, 0.80),
}
MAIN_COLORS = ['red', 'green', 'yellow', 'white', 'lightblue']

# ---- the model list: (name, kind, nx, ny, colour, is_main, in_lab) ----
SPECS = []
for c in MAIN_COLORS:
    SPECS.append((f'block_2x4_{c}', 'brick', 4, 2, c, True, True))
# lab distractors
for c in ['red', 'green', 'yellow', 'white', 'skyblue', 'black']:
    SPECS.append((f'block_2x2_{c}', 'brick', 2, 2, c, False, True))
for c in ['white', 'blue']:
    SPECS.append((f'block_2x6_{c}', 'brick', 6, 2, c, False, True))
for c in ['lime', 'blue', 'orange']:
    SPECS.append((f'block_2x4_{c}', 'brick', 4, 2, c, False, True))
SPECS.append(('block_4x4_yellow', 'brick', 4, 4, 'yellow', False, True))
SPECS.append(('block_step_2x4_lime', 'step_both', 4, 2, 'lime', False, True))
SPECS.append(('block_step_2x3_paleyellow', 'step_one', 3, 2, 'paleyellow', False, True))
# sim-only distractors
for c in MAIN_COLORS:
    SPECS.append((f'block_nostud_2x4_{c}', 'nostud', 4, 2, c, False, False))
SPECS.append(('block_2x2_lightblue', 'brick', 2, 2, 'lightblue', False, False))
for c in ['red', 'green', 'yellow', 'lightblue']:
    SPECS.append((f'block_2x6_{c}', 'brick', 6, 2, c, False, False))
for c in ['paleyellow', 'skyblue', 'cream']:
    SPECS.append((f'block_2x4_{c}', 'brick', 4, 2, c, False, False))


# ------------------------------------------------------------------ SDF pieces
def rgba(c, k=1.0):
    r, g, b = COLORS[c]
    return f'{r * k:.3f} {g * k:.3f} {b * k:.3f} 1'


def material(c, k=1.0):
    return (f'<material><ambient>{rgba(c, k)}</ambient><diffuse>{rgba(c, k)}</diffuse>'
            f'<specular>0.25 0.25 0.25 1</specular></material>')


SURFACE = (f'<surface><friction><ode><mu>{FRICTION}</mu><mu2>{FRICTION}</mu2></ode>'
           f'</friction></surface>')


def box(name, x, y, z, sx, sy, sz, c, k=1.0, collide=True):
    """Box centred at (x, y, z)."""
    geo = f'<geometry><box><size>{sx:.5f} {sy:.5f} {sz:.5f}</size></box></geometry>'
    pose = f'<pose>{x:.5f} {y:.5f} {z:.5f} 0 0 0</pose>'
    out = f'<visual name="{name}_v">{pose}{geo}{material(c, k)}</visual>'
    if collide:
        out += f'<collision name="{name}_c">{pose}{geo}{SURFACE}</collision>'
    return out


def cyl(name, x, y, z, r, length, c, k=1.0, collide=True):
    """Upright cylinder centred at (x, y, z)."""
    geo = f'<geometry><cylinder><radius>{r:.5f}</radius><length>{length:.5f}</length></cylinder></geometry>'
    pose = f'<pose>{x:.5f} {y:.5f} {z:.5f} 0 0 0</pose>'
    out = f'<visual name="{name}_v">{pose}{geo}{material(c, k)}</visual>'
    if collide:
        out += f'<collision name="{name}_c">{pose}{geo}</collision>'
    return out


def stud(name, x, y, z_base, c):
    """Hollow-looking Duplo stud standing on z_base: a cylinder with a darker inner disc on top."""
    out = cyl(name, x, y, z_base + STUD_H / 2, STUD_D / 2, STUD_H, c)
    out += cyl(name + '_ring', x, y, z_base + STUD_H + 0.0002, STUD_D / 2 * RING, 0.0004, c, k=0.72,
               collide=False)
    return out


def stud_xy(nx, ny):
    for i in range(nx):
        for j in range(ny):
            yield (i - (nx - 1) / 2) * PITCH, (j - (ny - 1) / 2) * PITCH


def parts_for(kind, nx, ny, c):
    lx, ly = nx * PITCH, ny * PITCH
    parts = []
    if kind in ('brick', 'nostud'):
        parts.append(box('body', 0, 0, BODY_H / 2, lx, ly, BODY_H, c))
        if kind == 'brick':
            for k, (x, y) in enumerate(stud_xy(nx, ny)):
                parts.append(stud(f'stud{k}', x, y, BODY_H, c))
        top = BODY_H + (STUD_H if kind == 'brick' else 0)

    elif kind in ('step_both', 'step_one'):
        # full-height middle part, lower step rows at the end(s), studs on both levels
        low_cols = [0, nx - 1] if kind == 'step_both' else [nx - 1]
        k = 0
        for i in range(nx):
            x = (i - (nx - 1) / 2) * PITCH
            h = STEP_H if i in low_cols else BODY_H
            parts.append(box(f'col{i}', x, 0, h / 2, PITCH, ly, h, c))
            for j in range(ny):
                y = (j - (ny - 1) / 2) * PITCH
                parts.append(stud(f'stud{k}', x, y, h, c))
                k += 1
        top = BODY_H + STUD_H

    elif kind == 'upside':
        # lying on its studs, open hollow side up: studs, top plate, 4 walls, 3 tubes
        for k, (x, y) in enumerate(stud_xy(nx, ny)):
            parts.append(cyl(f'stud{k}', x, y, STUD_H / 2, STUD_D / 2, STUD_H, c))
        z0 = STUD_H
        plate = 0.002
        parts.append(box('plate', 0, 0, z0 + plate / 2, lx, ly, plate, c))
        wh = BODY_H - plate
        zc = z0 + plate + wh / 2
        parts.append(box('wall_px', lx / 2 - WALL / 2, 0, zc, WALL, ly, wh, c))
        parts.append(box('wall_nx', -lx / 2 + WALL / 2, 0, zc, WALL, ly, wh, c))
        parts.append(box('wall_py', 0, ly / 2 - WALL / 2, zc, lx - 2 * WALL, WALL, wh, c))
        parts.append(box('wall_ny', 0, -ly / 2 + WALL / 2, zc, lx - 2 * WALL, WALL, wh, c))
        # inside is darker: a shaded floor, plus tubes between the stud rows
        parts.append(box('inside', 0, 0, z0 + plate + 0.0002, lx - 2 * WALL, ly - 2 * WALL, 0.0004, c,
                         k=0.55, collide=False))
        for t in range(nx - 1):
            x = (t - (nx - 2) / 2) * PITCH
            parts.append(cyl(f'tube{t}', x, 0, zc, 0.0066, wh, c, k=0.85, collide=False))
            parts.append(cyl(f'tube{t}_in', x, 0, z0 + BODY_H + 0.0002, 0.0050, 0.0004, c, k=0.5,
                             collide=False))
        top = STUD_H + BODY_H
    else:
        raise ValueError(kind)
    return parts, lx, ly, top


def make_sdf(name, kind, nx, ny, c):
    parts, lx, ly, top = parts_for(kind, nx, ny, c)
    mass = MASS_2X4 * (nx * ny) / 8.0
    h = top
    ixx = mass / 12 * (ly ** 2 + h ** 2)
    iyy = mass / 12 * (lx ** 2 + h ** 2)
    izz = mass / 12 * (lx ** 2 + ly ** 2)
    body = '\n        '.join(parts)
    sdf = f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{name}">
    <link name="link">
      <inertial>
        <pose>0 0 {h / 2:.5f} 0 0 0</pose>
        <mass>{mass:.5f}</mass>
        <inertia><ixx>{ixx:.4e}</ixx><iyy>{iyy:.4e}</iyy><izz>{izz:.4e}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
        {body}
    </link>
  </model>
</sdf>
"""
    return sdf, lx, ly, top


def make_config(name):
    return f"""<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>2.0</version>
  <sdf version="1.8">model.sdf</sdf>
  <description>Duplo-size block for the igus ReBeL pick-and-place simulation.</description>
</model>
"""


def main():
    out = os.path.abspath(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else 'models'))
    os.makedirs(out, exist_ok=True)
    manifest = []
    for name, kind, nx, ny, c, is_main, in_lab in SPECS:
        sdf, lx, ly, top = make_sdf(name, kind, nx, ny, c)
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'model.sdf'), 'w') as f:
            f.write(sdf)
        with open(os.path.join(d, 'model.config'), 'w') as f:
            f.write(make_config(name))
        manifest.append({'model': name, 'kind': kind, 'studs_x': nx, 'studs_y': ny,
                         'color': c, 'is_main': is_main, 'in_lab': in_lab,
                         'length': round(lx, 5), 'width': round(ly, 5), 'height': round(top, 5)})
    with open(os.path.join(out, 'blocks_manifest.json'), 'w') as f:
        json.dump({'main_colors': MAIN_COLORS, 'blocks': manifest}, f, indent=1)
    n_main = sum(m['is_main'] for m in manifest)
    print(f'wrote {len(manifest)} models ({n_main} main, {len(manifest) - n_main} distractors) to {out}')


if __name__ == '__main__':
    main()
