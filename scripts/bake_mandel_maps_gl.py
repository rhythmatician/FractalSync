import json
from pathlib import Path

import numpy as np
import moderngl


RE_MIN, RE_MAX = -2.0, 1.0
IM_MIN, IM_MAX = -1.5, 1.5

W0 = 2048
H0 = 2048

MAX_ITER = 512
R = 2.0
G0 = 0.02  # S = G/(G+G0)


VS = r"""
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = (in_pos + 1.0) * 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FS_F = r"""
#version 330
in vec2 v_uv;
out float out_F;

uniform float u_re_min;
uniform float u_re_max;
uniform float u_im_min;
uniform float u_im_max;

uniform int   u_max_iter;
uniform float u_R;

void main() {
    float cre = mix(u_re_min, u_re_max, v_uv.x);
    float cim = mix(u_im_min, u_im_max, v_uv.y);
    vec2 c = vec2(cre, cim);

    vec2 z = vec2(0.0, 0.0);
    float R2 = u_R * u_R;

    int n_escape = -1;
    float zabs = 1.0;

    for (int n = 0; n < 4096; n++) {
        if (n >= u_max_iter) break;

        // z = z^2 + c
        float x = z.x;
        float y = z.y;
        z = vec2(x*x - y*y, 2.0*x*y) + c;

        float mag2 = dot(z, z);
        if (mag2 > R2) {
            n_escape = n;
            zabs = sqrt(mag2);
            break;
        }
    }

    float nu;
    if (n_escape < 0) {
        nu = float(u_max_iter);
    } else {
        // nu = n + 1 - log2(log(|z|))
        // GLSL log is natural log.
        float t = log(log(zabs)) / log(2.0);
        nu = float(n_escape) + 1.0 - t;
        if (nu < 0.0) nu = 0.0;
        if (nu > float(u_max_iter)) nu = float(u_max_iter);
    }

    out_F = clamp(nu / float(u_max_iter), 0.0, 1.0);
}
"""

FS_S = r"""
#version 330
in vec2 v_uv;
out float out_S;

uniform sampler2D u_F_tex;
uniform int u_level;

uniform float u_re_min;
uniform float u_re_max;
uniform float u_im_min;
uniform float u_im_max;

uniform float u_G0;

void main() {
    ivec2 size = textureSize(u_F_tex, u_level);

    // Convert v_uv to nearest texel center
    int ix = int(floor(v_uv.x * float(size.x) + 0.5));
    int iy = int(floor(v_uv.y * float(size.y) + 0.5));
    ix = clamp(ix, 0, size.x - 1);
    iy = clamp(iy, 0, size.y - 1);

    int ixm = max(ix - 1, 0);
    int ixp = min(ix + 1, size.x - 1);
    int iym = max(iy - 1, 0);
    int iyp = min(iy + 1, size.y - 1);

    float F_xp = texelFetch(u_F_tex, ivec2(ixp, iy), u_level).r;
    float F_xm = texelFetch(u_F_tex, ivec2(ixm, iy), u_level).r;
    float F_yp = texelFetch(u_F_tex, ivec2(ix, iyp), u_level).r;
    float F_ym = texelFetch(u_F_tex, ivec2(ix, iym), u_level).r;

    // c-plane spacing at this mip level
    float dRe = (u_re_max - u_re_min) / float(max(size.x - 1, 1));
    float dIm = (u_im_max - u_im_min) / float(max(size.y - 1, 1));

    float dF_dRe = (F_xp - F_xm) / (2.0 * dRe);
    float dF_dIm = (F_yp - F_ym) / (2.0 * dIm);

    float G = sqrt(dF_dRe*dF_dRe + dF_dIm*dF_dIm);
    float S = G / (G + u_G0);

    out_S = clamp(S, 0.0, 1.0);
}
"""


def tex_read_f32(tex: moderngl.Texture, level: int) -> np.ndarray:
    w, h = tex.size
    w_l = max(w >> level, 1)
    h_l = max(h >> level, 1)
    raw = tex.read(level=level, alignment=4)
    arr = np.frombuffer(raw, dtype=np.float32).reshape((h_l, w_l))
    # moderngl/OpenGL origin is bottom-left; flip so row 0 corresponds to IM_MAX (top)
    return np.flipud(arr)


def write_mips_bin(mips: list[np.ndarray], out_path: Path) -> dict:
    offsets = []
    widths = []
    heights = []
    buf = bytearray()
    offset = 0

    for mip in mips:
        h, w = mip.shape
        widths.append(int(w))
        heights.append(int(h))
        offsets.append(int(offset))
        raw = mip.astype("<f4", copy=False).tobytes(order="C")
        buf.extend(raw)
        offset += len(raw)

    out_path.write_bytes(buf)
    return {
        "mip_widths": widths,
        "mip_heights": heights,
        "mip_offsets_bytes": offsets,
        "mip_levels": len(mips),
    }


def main():
    ctx = moderngl.create_standalone_context(require=330)

    # Fullscreen quad
    vbo = ctx.buffer(
        np.array(
            [
                -1.0,
                -1.0,
                1.0,
                -1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        ).tobytes()
    )
    vao_content = [(vbo, "2f", "in_pos")]

    # 1) Render F0 to R32F texture and build mipmaps
    texF = ctx.texture((W0, H0), components=1, dtype="f4")
    texF.filter = (moderngl.NEAREST, moderngl.NEAREST)
    fboF = ctx.framebuffer(color_attachments=[texF])

    progF = ctx.program(vertex_shader=VS, fragment_shader=FS_F)
    vaoF = ctx.vertex_array(progF, vao_content)

    progF["u_re_min"].value = RE_MIN
    progF["u_re_max"].value = RE_MAX
    progF["u_im_min"].value = IM_MIN
    progF["u_im_max"].value = IM_MAX
    progF["u_max_iter"].value = MAX_ITER
    progF["u_R"].value = R

    fboF.use()
    ctx.disable(moderngl.DEPTH_TEST)
    vaoF.render(mode=moderngl.TRIANGLES)

    # Generate mip pyramid for F
    texF.build_mipmaps()

    # Read back all mip levels for F
    mip_levels = int(np.log2(W0)) + 1  # 0..11 for 2048
    F_mips = [tex_read_f32(texF, level=i) for i in range(mip_levels)]

    # 2) For each mip level, render S at that mip's resolution and collect S_mips.
    # We compute S per mip using texelFetch at that mip level (so it's scale-consistent).
    progS = ctx.program(vertex_shader=VS, fragment_shader=FS_S)
    vaoS = ctx.vertex_array(progS, vao_content)

    progS["u_re_min"].value = RE_MIN
    progS["u_re_max"].value = RE_MAX
    progS["u_im_min"].value = IM_MIN
    progS["u_im_max"].value = IM_MAX
    progS["u_G0"].value = G0

    texF.use(location=0)
    progS["u_F_tex"].value = 0

    S_mips = []
    for level in range(mip_levels):
        w_l = max(W0 >> level, 1)
        h_l = max(H0 >> level, 1)

        texS = ctx.texture((w_l, h_l), components=1, dtype="f4")
        texS.filter = (moderngl.NEAREST, moderngl.NEAREST)
        fboS = ctx.framebuffer(color_attachments=[texS])

        progS["u_level"].value = level

        fboS.use()
        vaoS.render(mode=moderngl.TRIANGLES)

        # readback level 0 of texS (it has no mips; it *is* the per-mip S)
        raw = texS.read(alignment=4)
        arr = np.frombuffer(raw, dtype=np.float32).reshape((h_l, w_l))
        S_mips.append(np.flipud(arr))

    # Write assets
    out_F = Path("mandel_F_mips_f32.bin")
    out_S = Path("mandel_S_mips_f32.bin")
    meta_F = write_mips_bin(F_mips, out_F)
    meta_S = write_mips_bin(S_mips, out_S)

    meta = {
        "re_min": RE_MIN,
        "re_max": RE_MAX,
        "im_min": IM_MIN,
        "im_max": IM_MAX,
        "W0": W0,
        "H0": H0,
        "max_iter": MAX_ITER,
        "escape_radius": R,
        "G0": G0,
        "dtype": "f32_le_rowmajor",
        "F": meta_F,
        "S": meta_S,
        "note": "Arrays are stored row-major with row 0 corresponding to IM_MAX due to flipud on readback.",
    }
    Path("mandel_mips_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("Wrote:", out_F, out_F.stat().st_size / 1e6, "MB")
    print("Wrote:", out_S, out_S.stat().st_size / 1e6, "MB")
    print("Wrote: mandel_mips_meta.json")


if __name__ == "__main__":
    main()
