{
  description = "moonai flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
  let
    name = "moonai-flake";

    pkgs-common = with pkgs; [
      just
    ];

    pkgs-dev = with pkgs; [
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
      pkg-config
    ];

    pkgs-docs = with pkgs; [
      uv
    ];

    pkgs-shell-default = pkgs-common ++ pkgs-dev ++ pkgs-docs;
    pkgs-shell-docs = pkgs-common ++ pkgs-dev;

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
    devShells.${system} = {
      default = pkgs.mkShell {
        inherit name;
        inherit env;
        inherit shellHook;
        packages = pkgs-shell-default;
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath pkgs-shell-default;
      };

      ci-docs = pkgs.mkShell {
        packages = pkgs-shell-docs;
      };
    };
  };
}
