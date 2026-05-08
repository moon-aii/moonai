{
  description = "moonai flake environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: 
  let
    name = "moonai-shell";

    libs = with pkgs; [
      cudatoolkit
      libx11 libxi libxrandr libxcursor libGL libGLU
      udev
      zlib
      openssl
      stdenv.cc.cc.lib
    ];

    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
      };
    };
  in {
    devShells.${system}.default = pkgs.mkShell {
      inherit name;
      strictDeps = true;
      buildInputs = libs;
      NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;

      packages = with pkgs; [
        clang-tools
        cudatoolkit
        mermaid-cli
        texliveFull
        prettier
        bun
        uv
        rustup
        pkg-config
      ];

      env = {
        CUDA_PATH = "${pkgs.cudatoolkit}";
      };

      shellHook = ''
        echo "- ${name} dev shell activated."
      '';
    };
  };
}
