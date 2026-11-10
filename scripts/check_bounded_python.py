"""Exercise the opt-in Python binding using a fully synthetic prepared asset."""

import json
import struct

from runtime_core import BoundedOperationalRollout


def main() -> None:
    asset = b"FSOP0001" + struct.pack("<dddQQQ", -0.5, -0.5, 0.1, 12, 12, 0)
    asset += struct.pack("<144d", *([1.0] * 144))
    field_id = BoundedOperationalRollout.asset_field_id(asset)
    policy = json.dumps({"field_id": field_id, "refusals": []})
    for kappa, mu in ((-0.1, 0.0), (0.0, -0.1)):
        try:
            BoundedOperationalRollout(
                asset, policy, 0.1, 0.02, 0.0, kappa, mu, 0.01, 0.0
            )
        except ValueError as error:
            assert "invalid experimental Physics profile" in str(error)
        else:
            raise AssertionError("negative potential coefficient must refuse")
    engine = BoundedOperationalRollout(
        asset, policy, 0.1, 0.02, 0.0, 0.0, 0.0, 0.01, 0.0
    )
    initial = json.loads(engine.start(0.0, 0.0, 0.1, 0.0))
    result = json.loads(engine.step(0.0, 0.0))
    assert result["field_id"] == initial["field_id"] == engine.field_id()
    assert (
        result["physics_profile_id"]
        == initial["physics_profile_id"]
        == engine.physics_profile_id()
    )
    assert result["mechanical_time"] == 0.01
    before = engine.state()
    try:
        engine.step(1e300, 0.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("unavailable or nonfinite stage must refuse")
    assert engine.state() == before
    cut = json.dumps(
        {
            "field_id": field_id,
            "refusals": [
                {
                    "bounds": [-0.01, -0.01, 0.01, 0.01],
                    "authority": "synthetic reference cut",
                }
            ],
        }
    )
    blocked = BoundedOperationalRollout(asset, cut, 0.1, 0.02, 0.0, 0.0, 0.0, 0.01, 0.0)
    try:
        blocked.start(0.0, 0.0, 0.0, 0.0)
    except RuntimeError as error:
        assert "ReferenceSingularity" in str(error)
    else:
        raise AssertionError("authoritative reference cut must refuse")
    print(
        json.dumps(
            {
                "field_id": field_id,
                "physics_profile_id": engine.physics_profile_id(),
                "state": result,
                "transaction_and_reference_refusal": "passed",
            }
        )
    )


if __name__ == "__main__":
    main()
