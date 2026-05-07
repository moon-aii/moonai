use std::path::PathBuf;

fn emit_cuda_link_search_paths() {
    let cuda_root = std::env::var_os("CUDA_HOME")
        .or_else(|| std::env::var_os("CUDA_PATH"))
        .or_else(|| std::env::var_os("CUDAToolkit_ROOT"));

    let Some(cuda_root) = cuda_root else {
        return;
    };

    let cuda_root = PathBuf::from(cuda_root);
    for candidate in [cuda_root.join("lib64"), cuda_root.join("lib")] {
        if candidate.exists() {
            println!("cargo:rustc-link-search=native={}", candidate.display());
        }
    }
}

fn main() {
    emit_cuda_link_search_paths();
    println!("cargo:rerun-if-changed=src/tick/kernel.cu");
    println!("cargo:rerun-if-changed=src/tick/crossover.cu");
    println!("cargo:rerun-if-changed=src/tick/mutation.cu");
    println!("cargo:rerun-if-changed=src/tick/network_compilation.cu");

    cc::Build::new()
        .cuda(true)
        .flag("-arch=native")
        .file("src/tick/kernel.cu")
        .compile("moonai_simulation_cuda");

    cc::Build::new()
        .cuda(true)
        .flag("-arch=native")
        .files(&["src/tick/crossover.cu", "src/tick/mutation.cu", "src/tick/network_compilation.cu"])
        .compile("moonai_evolution_cuda");
}
