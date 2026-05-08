use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rustc-link-lib=dylib=cudart");

    println!("cargo:rerun-if-changed=src/tick/evolution_cuda.cuh");
    println!("cargo:rerun-if-changed=src/tick/kernel.cu");
    println!("cargo:rerun-if-changed=src/tick/crossover.cu");
    println!("cargo:rerun-if-changed=src/tick/mutation.cu");
    println!("cargo:rerun-if-changed=src/tick/network_compilation.cu");

    cc::Build::new()
        .cuda(true)
        .flag("-arch=native")
        .flag("-O2")
        .files(&[
            "src/tick/kernel.cu",
            "src/tick/crossover.cu",
            "src/tick/mutation.cu",
            "src/tick/network_compilation.cu",
        ])
        .compile("moonai_cuda");
}
