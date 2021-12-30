use std::env;
fn main() {
    println!("cargo:rerun-if-changed=cu/tools.cu");

    match env::var("NVCC") {
        Ok(var) => which::which(var),
        Err(_) => which::which("nvcc"),
    }
    .unwrap();

    cc::Build::new()
        .cuda(true)
        .cudart("static")
        .file("cu/tools.cu")
        .compile("libcutools.a");
}
