{
  description = "moonai flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
  let
    name = "moonai-flake";

    packages = with pkgs; [
      clang-tools
      wayland
      libxkbcommon
      vulkan-loader
      libGL
      cudatoolkit

      mermaid-cli
      texliveFull
      bun
      uv
      rustup
      just
      pkg-config
    ];

    env = {
      CUDA_PATH = "${pkgs.cudatoolkit}";
    };

    shellHook = ''
      echo "- ${name} shell activated."
    '';

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
      inherit packages;
      inherit env;
      inherit shellHook;
      LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath packages;
    };
  };
}
