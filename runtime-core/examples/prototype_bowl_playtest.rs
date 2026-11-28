//! Local, opt-in exterior bowl playtest.
use num_complex::Complex64;
use runtime_core::distance_field::{load_builtin_distance_field, sample_signed_distance_field};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

const DT: f64 = 1.0 / 512.0;
const RASTER: [[f64; 2]; 2] = [[-2.0, -1.122], [0.4711, 1.122]];

fn log_scale(x: f64, y: f64) -> f64 {
    let margin = (x - RASTER[0][0])
        .min(RASTER[1][0] - x)
        .min(y - RASTER[0][1])
        .min(RASTER[1][1] - y);
    // Smooth the four edge fades independently to avoid corner kinks.
    if margin <= 0.04 {
        return 0.0;
    }
    let fade = |d: f64| {
        let t = ((d - 0.04) / 0.20).clamp(0.0, 1.0);
        t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
    };
    let weight = fade(x - RASTER[0][0])
        * fade(RASTER[1][0] - x)
        * fade(y - RASTER[0][1])
        * fade(RASTER[1][1] - y);
    let d = f64::from(
        sample_signed_distance_field(&[Complex64::new(x, y)]).expect("built-in raster")[0],
    );
    -weight * d.hypot(0.15).ln()
}

fn potential(x: f64, y: f64) -> (f64, f64) {
    let r = x.hypot(y);
    if r <= 2.0 {
        return (0.0, 0.0);
    }
    let q = ((r - 2.0) / 2.0).ln_1p();
    (4.0 * q.powi(4), -16.0 * q.powi(3) / (r * r))
}

fn rhs([x, y, v, w]: [f64; 4], [ax, ay]: [f64; 2]) -> [f64; 4] {
    let h = 0.0005;
    let gx = (log_scale(x + h, y) - log_scale(x - h, y)) / (2.0 * h);
    let gy = (log_scale(x, y + h) - log_scale(x, y - h)) / (2.0 * h);
    let f = potential(x, y).1;
    let inv = (-2.0 * log_scale(x, y)).exp();
    let speed = v * v + w * w;
    let dot = gx * v + gy * w;
    [
        v,
        w,
        speed * gx - 2.0 * dot * v + inv * (f * x + ax),
        speed * gy - 2.0 * dot * w + inv * (f * y + ay),
    ]
}

fn step(s: [f64; 4], drive: [f64; 2], dt: f64) -> [f64; 4] {
    let a = rhs(s, drive);
    let b = rhs(std::array::from_fn(|i| s[i] + dt * a[i] / 2.0), drive);
    let c = rhs(std::array::from_fn(|i| s[i] + dt * b[i] / 2.0), drive);
    let d = rhs(std::array::from_fn(|i| s[i] + dt * c[i]), drive);
    std::array::from_fn(|i| s[i] + dt * (a[i] + 2.0 * b[i] + 2.0 * c[i] + d[i]) / 6.0)
}

fn energy([x, y, v, w]: [f64; 4]) -> f64 {
    (2.0 * log_scale(x, y)).exp() * (v * v + w * w) / 2.0 + potential(x, y).0
}

fn launch(tangent: bool) -> [f64; 4] {
    if tangent {
        [0.0, 0.9, 0.22, 0.30]
    } else {
        [0.0, 0.9, 0.0, 0.42]
    }
}

fn check() {
    for dt in [DT, DT / 2.0] {
        for tangent in [false, true] {
            let mut s = launch(tangent);
            let initial = energy(s);
            let mut crossed = false;
            let mut returned = false;
            let mut peak: f64 = 0.0;
            let mut error: f64 = 0.0;
            for _ in 0..(24.0 / dt) as usize {
                s = step(s, [0.0; 2], dt);
                let r = s[0].hypot(s[1]);
                peak = peak.max(r);
                error = error.max((energy(s) - initial).abs());
                if r > 2.0 {
                    crossed = true;
                } else if crossed {
                    returned = true;
                    break;
                }
            }
            println!(
                "dt={dt:.6} tangent={tangent} peak={peak:.6} returned={returned} energy_error={error:.6e}"
            );
            assert!(returned && error / initial < 0.03);
        }
    }
}

fn reply(mut client: TcpStream, state: &mut [f64; 4]) -> std::io::Result<()> {
    client.set_read_timeout(Some(std::time::Duration::from_secs(2)))?;
    let mut request = [0; 4096];
    let n = client.read(&mut request)?;
    let request = String::from_utf8_lossy(&request[..n]);
    let path = request.split_whitespace().nth(1).unwrap_or("");
    let (kind, body) = if path == "/" {
        (
            "text/html; charset=utf-8",
            include_str!("prototype_bowl_playtest.html").to_owned(),
        )
    } else if path.starts_with("/launch/") || path.starts_with("/tick/") {
        if path.starts_with("/launch/") {
            *state = launch(path.ends_with("tangent"));
        }
        if let Some(input) = path.strip_prefix("/tick/") {
            let mut xy = input
                .split('/')
                .map(|axis| axis.parse::<i32>().unwrap_or(0).clamp(-1, 1));
            let drive = [
                f64::from(xy.next().unwrap_or(0)) * 0.12,
                f64::from(xy.next().unwrap_or(0)) * 0.12,
            ];
            for _ in 0..8 {
                let next = step(*state, drive, DT);
                if next.into_iter().all(f64::is_finite) {
                    *state = next;
                }
            }
        }
        ("application/json", serde_json::json!({"x": state[0], "y": state[1], "radius": state[0].hypot(state[1]), "energy": energy(*state)}).to_string())
    } else {
        return write!(
            client,
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
    };
    write!(
        client,
        "HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn main() {
    load_builtin_distance_field("default").unwrap();
    if std::env::args().any(|arg| arg == "--check") {
        check();
        return;
    }
    let listener = TcpListener::bind("127.0.0.1:4178").unwrap();
    let mut state = launch(false);
    println!("Bowl playtest at http://127.0.0.1:4178/");
    for stream in listener.incoming().flatten() {
        let _ = reply(stream, &mut state);
    }
}
