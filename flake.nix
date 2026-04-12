{
  description = "moonai nix flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable"; 
  };

  outputs = { self, nixpkgs }:
  let
    supportedSystems = [
      "x86_64-linux"
    ];
    forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
  in {
    devShells = forAllSystems (system: 
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
      in
      {
        default = pkgs.mkShell {
          packages = with pkgs; [
            cmake
            ninja
            gnumake
            just
            gcc
            vcpkg
            clang-tools
            cppcheck
            llvm
            pkg-config
            binutils

            uv
            python312
            pylyzer

            fontconfig
            expat

            texliveFull
            plantuml
            graphviz
            corefonts

            cudatoolkit
            libx11
            libxi
            libxrandr
            libxcursor
            udev
            libGL
            libGLU
          ];

          AR = "${pkgs.gcc}/bin/gcc-ar";
          RANLIB = "${pkgs.gcc}/bin/gcc-ranlib";
          CC = "${pkgs.gcc}/bin/gcc";
          CXX = "${pkgs.gcc}/bin/g++";
          VCPKG_ROOT = "${pkgs.vcpkg}/share/vcpkg";
          CUDA_PATH = "${pkgs.cudatoolkit}";
          LD_LIBRARY_PATH = "/run/opengl-driver/lib:${pkgs.cudatoolkit}/lib:${
            pkgs.lib.makeLibraryPath [ 
              pkgs.libGL 
              pkgs.libGLU 
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
            ]
          }";

          UV_PYTHON_DOWNLOADS = "never";
          UV_PYTHON = "${pkgs.python312}/bin/python3";

          FONTCONFIG_FILE = pkgs.makeFontsConf {
            fontDirectories = [ pkgs.corefonts ];
          };

          shellHook = ''
            echo "Project Packages and environment loaded for ${system}."
          '';
        };
      }
    );
  };
}
