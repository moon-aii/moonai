{ pkgs ? import <nixpkgs> {
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
  }
}:

let
  name = "moonai-shell";

  libs = with pkgs; [
    stdenv.cc.cc.lib
    zlib
    openssl
    udev

    cudaPackages.cudatoolkit

    libx11
    libxi
    libxrandr
    libxcursor
    libGL
    libGLU
  ];
in
pkgs.mkShell {
  name = name;
  strictDeps = true;
  buildInputs = libs;

  packages = with pkgs; [
    pkg-config
    stdenv.cc

    cudaPackages.cudatoolkit
    clang-tools

    prettier
    bun
    uv
    rustup
    just
  ];

  env = {
    CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
    NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
  };

  shellHook = ''
    echo "- ${name} dev shell activated."
  '';
}
