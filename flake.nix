{
  description = "moonai flake environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: 
  let
    name = "moonai-flake";

    libs = with pkgs; [
      wayland
      libxkbcommon
      vulkan-loader
      libGL
      cudatoolkit
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
      LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;

      packages = with pkgs; [
        clang-tools
        cudatoolkit
        mermaid-cli
        texliveFull
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
