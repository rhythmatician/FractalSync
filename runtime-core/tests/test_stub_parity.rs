use std::fs;
use std::path::PathBuf;

#[test]
fn stub_functions_exported_in_pybindings() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let stub_path = manifest.join("runtime_core.pyi");
    let bindings_path = manifest.join("src").join("pybindings.rs");

    let stub_src = fs::read_to_string(&stub_path)
        .expect(&format!("Failed to read stub file: {}", stub_path.display()));
    let bindings_src = fs::read_to_string(&bindings_path)
        .expect(&format!("Failed to read bindings file: {}", bindings_path.display()));

    // Collect top-level (non-indented) function names from the stub file
    let mut funcs = Vec::new();
    for line in stub_src.lines() {
        // Only consider top-level defs (no leading whitespace)
        if line.starts_with("def ") {
            if let Some(rest) = line.split_whitespace().nth(1) {
                if let Some(idx) = rest.find('(') {
                    let name = &rest[..idx];
                    if !name.starts_with("_") {
                        funcs.push(name.to_string());
                    }
                }
            }
        }
    }

    // Verify each function is exported from pybindings.rs either by wrapping or by a #[pyfunction]
    let mut missing = Vec::new();
    for f in funcs {
        let wrap_pat = format!("wrap_pyfunction!({},", f); // sometimes there are extra args
        let wrap_pat2 = format!("wrap_pyfunction!({} ", f);
        let pyfunc_pat = format!("#[pyfunction]\nfn {}(", f);
        if !bindings_src.contains(&wrap_pat) && !bindings_src.contains(&wrap_pat2) && !bindings_src.contains(&pyfunc_pat) {
            missing.push(f);
        }
    }

    assert!(missing.is_empty(), "Missing exports for functions: {:?}", missing);
}