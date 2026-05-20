use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only run stub generation when the `python` feature is enabled.
    if env::var("CARGO_FEATURE_PYTHON").is_ok() {
        // Destination directory for generated stubs (relative to repo root)
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let out_dir = manifest_dir.join("..").join("backend").join("stubs").join("runtime_core");
        let _ = std::fs::create_dir_all(&out_dir);

        // Attempt to invoke the Python-side generator. This avoids requiring
        // a Rust dependency and uses the developer's Python environment.
        // Note: keep the command simple and avoid shell features so it's portable.
        let status = Command::new("python")
            .arg("-m")
            .arg("pyo3_stubgen.generate")
            .arg("runtime_core")
            .arg("-o")
            .arg(out_dir.as_os_str())
            .arg("--package-name")
            .arg("runtime_core")
            .status();

        match status {
            Ok(s) if s.success() => println!("cargo:warning=runtime_core pyi generated"),
            Ok(s) => println!("cargo:warning=runtime_core pyi generator exited with status {}", s),
            Err(e) => println!("cargo:warning=runtime_core pyi generator failed: {}", e),
        }
    }
}
