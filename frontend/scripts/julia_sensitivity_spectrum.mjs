export function dot(a, b) {
  return a.reduce((sum, value, i) => sum + value * b[i], 0);
}

export function sensitivitySpectrum([v, w]) {
  const [a, b, d] = [dot(v, v), dot(v, w), dot(w, w)];
  const angle = Math.atan2(2 * b, a - d) / 2;
  const major = (a + d + Math.hypot(a - d, 2 * b)) / 2;
  const minor = major > 0 ? (a * d - b * b) / major : 0;
  const axis = [Math.cos(angle), Math.sin(angle)];
  return { gram: [[a, b], [b, d]], eigenvalues: [minor, major],
    eigenvectors_intrinsic: [[-axis[1], axis[0]], axis] };
}
