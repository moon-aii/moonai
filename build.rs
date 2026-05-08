use std::path::{Path, PathBuf};

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

fn search_path_for(tool_name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for entry in std::env::split_paths(&path_var) {
        let candidate = entry.join(tool_name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn sibling_tool(toolchain_path: &Path, tool_name: &str) -> Option<PathBuf> {
    let parent = toolchain_path.parent()?;
    let candidate = parent.join(tool_name);
    if candidate.is_file() { Some(candidate) } else { None }
}

fn resolve_archiver() -> Option<PathBuf> {
    std::env::var_os("AR").map(PathBuf::from).filter(|path| path.is_file()).or_else(|| search_path_for("ar")).or_else(
        || {
            ["CXX", "CC", "HOST_CXX", "HOST_CC"]
                .iter()
                .filter_map(std::env::var_os)
                .map(PathBuf::from)
                .find_map(|toolchain| sibling_tool(&toolchain, "ar"))
        },
    )
}

fn main() {
    println!("cargo:rerun-if-changed=src/tick/evolution_cuda.cuh");
    emit_cuda_link_search_paths();
    println!("cargo:rerun-if-changed=src/tick/kernel.cu");
    println!("cargo:rerun-if-changed=src/tick/crossover.cu");
    println!("cargo:rerun-if-changed=src/tick/mutation.cu");
    println!("cargo:rerun-if-changed=src/tick/network_compilation.cu");

    let mut build = cc::Build::new();
    if let Some(archiver) = resolve_archiver() {
        build.archiver(archiver);
    }

    build
        .cuda(true)
        .flag("-arch=native")
        .files(&[
            "src/tick/kernel.cu",
            "src/tick/crossover.cu",
            "src/tick/mutation.cu",
            "src/tick/network_compilation.cu",
        ])
        .compile("moonai_cuda");
}
