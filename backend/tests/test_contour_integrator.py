import runtime_core as rc


def test_contour_step_tangential_preserved():
    # 3x3 field with gradient in +x
    field = [
        0.0,
        0.5,
        1.0,
        0.0,
        0.5,
        1.0,
        0.0,
        0.5,
        1.0,
    ]
    df = rc.DistanceField(field, 3, (-1.0, 1.0), (-1.0, 1.0), 1.0, 0.1)

    real, imag = 0.0, 0.0

    # Proposed motion purely in +y (tangential), h=0 -> should largely be preserved
    out = rc.contour_biased_step(real, imag, 0.0, 0.5, 0.0, 0.5, 0.2, df)

    assert abs(out.imag - imag) > 0.0
    assert abs(out.real - real) < 0.1


def test_contour_step_normal_change_with_hit():
    field = [
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    df = rc.DistanceField(field, 3, (-1.0, 1.0), (-1.0, 1.0), 1.0, 0.1)
    real, imag = 0.0, 0.0

    # Proposed normal outward (x direction)
    out_no_hit = rc.contour_biased_step(real, imag, 0.1, 0.0, 0.0, 0.5, 1.0, df)
    out_hit = rc.contour_biased_step(real, imag, 0.1, 0.0, 1.0, 0.5, 1.0, df)

    dx_no_hit = abs(out_no_hit.real - real)
    dx_hit = abs(out_hit.real - real)
    assert dx_hit > dx_no_hit
