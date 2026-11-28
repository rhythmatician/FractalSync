"""Summarize the local aperture and similarity fits for captured Julia renders."""

import hashlib
from pathlib import Path
from PIL import Image
import numpy as np
from mpmath import iv
from scipy.ndimage import center_of_mass, gaussian_filter, label, map_coordinates
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2] / 'tmp' / 'julia-eyes'


def read(name):
    with Image.open(ROOT / f'{name}.png') as image:
        return np.asarray(image.convert('RGB'))

def aperture(img, center, threshold):
    """Return equivalent-area radius and pixel centroid of the marked opening."""
    x, y = center
    cut = img[y-48:y+49, x-48:x+49]
    labels, _ = label(cut.max(axis=2) <= threshold)
    k = labels[48, 48]
    if k == 0 or (labels[0] == k).any() or (labels[-1] == k).any() or (labels[:,0] == k).any() or (labels[:,-1] == k).any():
        return np.nan, np.full(2,np.nan)
    component = labels == k
    cy, cx = center_of_mass(component)
    return np.sqrt(component.sum() / np.pi), np.array([cx + x - 48, cy + y - 48])

def smooth(img):
    return gaussian_filter(img @ np.array([0.299, 0.587, 0.114]), 2.)

def fit(base, img, center, radii):
    """Fit a translation, rotation and scale from base pixels to target pixels."""
    x, y = center
    yy, xx = np.mgrid[-radii[1]:radii[1]+1, -radii[1]:radii[1]+1]
    keep = (xx**2+yy**2 >= radii[0]**2) & (xx**2+yy**2 <= radii[1]**2)
    u,v = xx[keep],yy[keep]
    ref = base[y+v,x+u]
    def residual(state):
        dx,dy,angle,logscale = state
        a,b = np.exp(logscale)*np.array([np.cos(angle),np.sin(angle)])
        return map_coordinates(img, [y+dy+b*u+a*v,x+dx+a*u-b*v], order=1, prefilter=False)-ref
    out = least_squares(residual, np.zeros(4), bounds=([-8,-8,-.3,-.25],[8,8,.3,.25]), x_scale=[1.,1.,.025,.025])
    if not out.success:
        raise RuntimeError(out.message)
    return out.x, np.linalg.norm(out.fun) / max(np.linalg.norm(residual(np.zeros(4))), 1e-12)

baseline = read('base')
assert np.array_equal(baseline,read('repeat'))
print('base PNG SHA256:',hashlib.sha256((ROOT/'base.png').read_bytes()).hexdigest())

# Use interval arithmetic for the decimal parameter. A double orbit diverges
# from this enclosure long before it escapes.
iv.dps = 300
parameter = iv.mpc(iv.mpf('-0.7269'), iv.mpf('0.1889'))
critical_orbit = iv.mpc(0, 0)
for iteration in range(1, 501):
    critical_orbit = critical_orbit * critical_orbit + parameter
    squared_modulus = critical_orbit.real**2 + critical_orbit.imag**2
    if squared_modulus.a > 4:
        print('Enclosed critical escape:', iteration, iv.nstr(squared_modulus, 16))
        break
    if not squared_modulus.b <= 4:
        raise RuntimeError(f'Unresolved interval at iteration {iteration}')
else:
    raise RuntimeError('No critical escape in the checked interval')

for mark in [(290,278),(509,321)]:
    print('mark',mark,'apertures',[(t,aperture(baseline,mark,t)) for t in (10,20,30)])
    for h in (.00025,.0005,.001):
        for axis in 'xy':
            lo, hi = [read(f'{axis}{sign}-{h}') for sign in 'mp']
            radii = [(np.log(aperture(hi,mark,t)[0])-np.log(aperture(lo,mark,t)[0]))/(2*h) for t in (10,20,30)]
            fits = []
            for annulus in [(15,30),(22,46),(30,55)]:
                p0,e0=fit(smooth(baseline),smooth(lo),mark,annulus)
                p1,e1=fit(smooth(baseline),smooth(hi),mark,annulus)
                angle,log_scale = ((p1-p0)/(2*h))[2:]
                fits.append((np.round(-angle,1),np.round(log_scale,1),np.round(max(e0,e1),2)))
            print(h,axis,np.round(radii,1),fits)
