{
  description = "cpp flake template";
  inputs = { nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable"; };
  outputs = { self, nixpkgs }:
  let
    supportedSystems = [ "x86_64-linux" ];
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
            gcc
            vcpkg
            clang-tools
            cppcheck
            llvm
            pkg-config
            binutils

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

          shellHook = ''
            echo "Project Packages and environment loaded for ${system}."
          '';
        };
      }
    );
  };
}
