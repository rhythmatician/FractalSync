use super::interval::Interval;
use super::certified_cardioid::{curve_box, distance_bounds};

pub(super) struct FeatureCover {
    query: [f64; 2],
    points: Vec<[Interval; 2]>,
    pub(super) visits: usize,
}
impl FeatureCover {
    pub(super) fn projections(&self)->Vec<[f64;2]> {
        self.points.iter().map(|p|[p[0].lo+(p[0].hi-p[0].lo)*0.5,
            p[1].lo+(p[1].hi-p[1].lo)*0.5]).collect()
    }
}

fn stationary(c: [f64; 2], a: f64, b: f64, cotangent: bool) -> Interval {
    let one=Interval::point(1.);
    let two=Interval::point(2.);
    let half=Interval::point(0.5);
    let s=Interval{lo:a,hi:b};
    let ss=s.square();
    let denominator=one.add(ss);
    let denominator2=denominator.square();
    let numerator=if cotangent {ss.sub(one)} else {one.sub(ss)};
    let u=numerator.div_positive(denominator);
    let v=two.mul(s).div_positive(denominator);
    let du=Interval::point(if cotangent {4.} else {-4.}).mul(s).div_positive(denominator2);
    let dv=two.mul(one.sub(ss)).div_positive(denominator2);
    let dx=half.mul(du).sub(half.mul(u.mul(du).sub(v.mul(dv))));
    let dy=half.mul(dv).sub(half.mul(du.mul(v).add(u.mul(dv))));
    let p=curve_box(a,b,cotangent);
    Interval::point(c[0]).sub(p[0]).mul(dx)
        .add(Interval::point(c[1]).sub(p[1]).mul(dy))
}

fn distance_lower(c:[f64;2], p:[Interval;2])->f64 {
    let x=Interval::point(c[0]).sub(p[0]);
    let y=Interval::point(c[1]).sub(p[1]);
    x.square().add(y.square()).lo.max(0.).sqrt().next_down().max(0.)
}

// Contains every global nearest point and every stationary point whose
// distance can lie within the supplied tie band. Boxes need not contain a
// root: retaining an extra candidate can only prevent a regular certificate.
// Budget limits feature subdivision. Distance preprocessing has its own cap.
pub(super) fn cover(c:[f64;2], tie_band:f64, spatial_width:f64, budget:usize)->Option<FeatureCover> {
    if budget==0 || !tie_band.is_finite() || tie_band<0. || !spatial_width.is_finite() || spatial_width<=0. {return None;}
    let bounds=distance_bounds(c,1e-13,8192)?;
    let threshold=(bounds.upper+tie_band).next_up();
    let mut stack=vec![(-1.,1.,false),(-1.,1.,true)];
    let mut points=Vec::new();
    let mut visits=0;
    while let Some((a,b,cotangent))=stack.pop() {
        visits+=1;
        if visits>budget {return None;}
        let point=curve_box(a,b,cotangent);
        if distance_lower(c,point)>threshold {continue;}
        let derivative=stationary(c,a,b,cotangent);
        if !derivative.finite() {return None;}
        if derivative.lo>0. || derivative.hi<0. {continue;}
        let width=(point[0].hi-point[0].lo).next_up()
            .max((point[1].hi-point[1].lo).next_up());
        if width<=spatial_width {points.push(point);continue;}
        let middle=a+(b-a)*0.5;
        if middle==a || middle==b {return None;}
        stack.push((middle,b,cotangent));
        stack.push((a,middle,cotangent));
    }
    (!points.is_empty()).then_some(FeatureCover{query:c,points,visits})
}

pub(super) fn directions_agree(cover:&FeatureCover, cosine:f64)->bool {
    let c=cover.query;
    if !cosine.is_finite() || !(-1. ..=1.).contains(&cosine) {return false;}
    let directions:Option<Vec<_>>=cover.points.iter().map(|p| {
        let v=[Interval::point(c[0]).sub(p[0]),Interval::point(c[1]).sub(p[1])];
        let square=v[0].square().add(v[1].square());
        let norm=Interval{lo:square.lo.max(0.).sqrt().next_down(),hi:square.hi.sqrt().next_up()};
        if !norm.finite() || norm.lo<=0. {return None;}
        Some([v[0].div_positive(norm),v[1].div_positive(norm)])
    }).collect();
    let Some(directions)=directions else {return false;};
    !directions.is_empty() && directions.iter().all(|a|directions.iter().all(|b|
        a[0].mul(b[0]).add(a[1].mul(b[1])).lo>=cosine))
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn regular_queries_have_complete_agreeing_direction_covers() {
        for c in [[-0.5,0.1],[0.,0.],[0.21474166530729011,0.12774018997553147],
            [0.32046083605181780,0.04736134021833255],[0.3,0.05]] {
            let cover=cover(c,0.5e-6,1e-6/64.,65536).expect("complete cover");
            assert!(directions_agree(&cover,30f64.to_radians().cos()),"{c:?}");
        }
    }
    #[test] fn symmetric_medial_axis_and_boundary_do_not_certify_regular() {
        for c in [[-0.25,0.],[0.25,0.],[-0.75,0.]] {
            let cover=cover(c,0.5e-6,1e-6/64.,65536).expect("complete cover");
            assert!(!directions_agree(&cover,30f64.to_radians().cos()),"{c:?}");
        }
    }
    #[test] fn candidate_cover_contains_the_cusp_at_origin() {
        let cover=cover([0.,0.],0.,1e-7,65536).unwrap();
        assert!(cover.points.iter().any(|p| p[0].lo<=0.25 && p[0].hi>=0.25 && p[1].lo<=0. && p[1].hi>=0.));
    }
    #[test] fn incomplete_or_invalid_cover_cannot_certify() {
        assert!(cover([0.,0.],0.,1e-7,0).is_none());
        assert!(cover([0.,0.],-1.,1e-7,65536).is_none());
        assert!(cover([f64::NAN,0.],0.,1e-7,65536).is_none());
        assert!(!directions_agree(&FeatureCover{query:[0.,0.],points:vec![],visits:0},0.8));
    }
    #[test] fn rational_chart_seams_retain_known_stationary_feet() {
        for sign in [-1.,1.] {
            let query=[0.20,sign*0.45];
            let cover=cover(query,0.5e-6,1e-6/64.,65536).unwrap();
            assert_eq!(cover.query,query);
            assert!(cover.points.iter().any(|p|p[0].lo<=0.25 && p[0].hi>=0.25 &&
                p[1].lo<=sign*0.5 && p[1].hi>=sign*0.5));
            assert!(directions_agree(&cover,30f64.to_radians().cos()));
        }
    }
    #[test] fn a_tie_band_near_the_medial_axis_keeps_both_directions() {
        for y in [-1e-10,0.,1e-10] {
            let cover=cover([-0.25,y],0.5e-6,1e-6/64.,65536).unwrap();
            assert!(cover.points.iter().any(|p|p[1].lo>0.));
            assert!(cover.points.iter().any(|p|p[1].hi<0.));
            assert!(!directions_agree(&cover,30f64.to_radians().cos()));
        }
    }
    #[test] fn equal_distance_opposite_feet_are_both_retained() {
        let cover=cover([-0.25,0.],0.,1e-6/64.,65536).unwrap();
        for x in [-0.75,0.25] {
            assert!(cover.points.iter().any(|p|p[0].lo<=x && p[0].hi>=x &&
                p[1].lo<=0. && p[1].hi>=0.));
        }
        assert!(!directions_agree(&cover,30f64.to_radians().cos()));
    }
}
