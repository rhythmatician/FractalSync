use std::sync::RwLock;
type Lazy<T>=std::sync::LazyLock<T>;
#[path="../../../runtime-core/src/geometry_provider/interval.rs"] mod interval;
#[path="../../../runtime-core/src/geometry_provider/uniform_box.rs"] mod uniform;
fn main() {
 let c=[0.32049965858459473_f64,0.04733508825302124_f64];
 for k in [20,24,26,27] {
  for stride in [1,8,32,128] {
  let h=2f64.powi(-(k as i32+3));let width=56.*h;let budget=512+64*k as usize;
  let origin=[(c[0]/h).floor()*h-width/2.,(c[1]/h).floor()*h-width/2.];
  let mut old_count=0;let mut new_count=0;let mut additional=0;let mut old_ns=0;let mut new_ns=0;
  for y in -4..=4 {for x in -4..=4 {
   let p=[origin[0]+(x*stride) as f64*width,origin[1]+(y*stride) as f64*width];
   let start=std::time::Instant::now();let old=uniform::membership(p[0],p[1],width);old_ns+=start.elapsed().as_nanos();
   let start=std::time::Instant::now();let new=uniform::membership_with_budget(p[0],p[1],width,budget);new_ns+=start.elapsed().as_nanos();
   old_count+=usize::from(old.is_some());new_count+=usize::from(new.is_some());
   if old.is_some() {assert_eq!(old,new);} else if new.is_some() {additional+=1;assert_eq!(new,Some(false));}
  }}
  println!("{{\"k\":{},\"stride\":{},\"budget\":{},\"boxes\":81,\"default_certified\":{},\"budget_certified\":{},\"new_exterior\":{},\"default_ns\":{},\"budget_ns\":{}}}",k,stride,budget,old_count,new_count,additional,old_ns,new_ns);
  }
 }
}
