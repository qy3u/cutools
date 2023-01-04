use std::env;
fn main() {
    let source_path = "cu/tools.cu";

    println!("cargo:rerun-if-changed={}", source_path);

    match env::var("NVCC") {
        Ok(var) => which::which(var),
        Err(_) => which::which("nvcc"),
    }
    .expect("nvcc not found");

    let mut builder = cc::Build::new();

    builder.cuda(true).cudart("static").file(&source_path);

    builder
        .flag("-arch=sm_86")
        .flag("--generate-code=arch=compute_86,code=sm_86")
        .flag("--generate-code=arch=compute_80,code=sm_80")
        .flag("--generate-code=arch=compute_75,code=sm_75")
        .flag("--generate-code=arch=compute_70,code=sm_70")
        .flag("--generate-code=arch=compute_61,code=sm_61");

    if cfg!(feature = "stream-per-thread") {
        builder.flag("--default-stream").flag("per-thread");
    }

    builder.compile("libcutools.a");
}
