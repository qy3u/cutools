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
        .flag("--gpu-architecture=sm_86")
        .flag("-gencode")
        .flag("arch=compute_86,code=sm_86")
        .flag("-gencode")
        .flag("arch=compute_80,code=sm_80")
        .flag("-gencode")
        .flag("arch=compute_75,code=sm_75")
        .flag("-gencode")
        .flag("arch=compute_70,code=sm_70")
        .flag("-gencode")
        .flag("arch=compute_60,code=sm_60");

    if cfg!(feature = "stream-per-thread") {
        builder.flag("--default-stream").flag("per-thread");
    }

    builder.compile("libcutools.a");
}
