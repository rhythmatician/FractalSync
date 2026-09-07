import { describe, it } from 'vitest';
import * as THREE from 'three';

describe('ThreeJS Matrix4.lookAt', () => {
  it('checks what Matrix4.lookAt does', () => {
    const eye = new THREE.Vector3(0, 0, 0);
    const target = new THREE.Vector3(0, 0, -1);
    const up = new THREE.Vector3(0, 1, 0);

    const m = new THREE.Matrix4().lookAt(eye, target, up);
    console.log('lookAt elements when looking down -Z:', m.elements);

    // If looking down -Z, m should be identity
    const p = new THREE.Vector3(0, 0, -1);
    p.applyMatrix4(m.clone().invert());
    console.log('transformed p:', p);

    // Now test looking at (0, -0.5, -1)
    const targetDown = new THREE.Vector3(0, -0.5, -1).normalize();
    const mDown = new THREE.Matrix4().lookAt(eye, targetDown, up);
    const pDown = targetDown.clone().applyMatrix4(mDown.clone().invert());
    console.log('transformed targetDown:', pDown);
  });
});
