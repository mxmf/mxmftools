{
  description = "Python development environment with uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    helix-erasin = {
      url = "github:erasin/helix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, ... } @inputs:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          system = system;
          # config.allowUnfree = true;
          # cudaSupport = true;
        };
        pkgs-unstable = import nixpkgs {
          # pkgs = import nixpkgs {
          system = system;
          # config.allowUnfree = true;
          # cudaSupport = true;
        };

      in
      {
        devShells.default = pkgs.mkShell rec{
          buildInputs = with pkgs; [
            nix

            uv
            basedpyright
            ruff

            # cudatoolkit
            # cudaPackages.cudatoolkit # CUDA Toolkit für GPU-Unterstützung
            # cudaPackages.cudnn # CUDA Deep Neural Network library
            # (pkgs-unstable.linuxPackages.nvidia_x11)


            #for libstdc++.so.6
            stdenv.cc.cc
            # for libz.so.1
            zlib

            python3Packages.pyqt5
            # python3Packages.ase
            python3Packages.matplotlib
            # python3Packages.tkinter
            qt5.qtwayland
            libsForQt5.qt5.qtbase
            # tk


          ];


          MPLBACKEND = "QtAgg";

          # QT_PLUGIN_PATH = "${pkgs.qt5.qtbase}/${pkgs.qt5.qtbase.qtPluginPrefix}";
          # QT_PLUGIN_PATH = "${pkgs.qt5.qtbase}/${pkgs.qt5.qtbase.qtPluginPrefix}";
          # QT_PLUGIN_PATH = ".venv/lib/python3.10/site-packages/PyQt5/Qt5/plugins";

          # LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
          # CUDA_PATH = pkgs.cudatoolkit;
          NVIDIA_DRIVERS = "/run/opengl-driver/lib";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (buildInputs ++ [ "/run/opengl-driver" ]);
          EXTRA_LDFLAGS = "-L/lib -L/run/opengl-driver/lib";


          shellHook = ''
            export ABACUS_PP_PATH="$HOME/.config/mxmftools/pp/sg15_oncv_upf_2020-02-06"
            export ABACUS_ORBITAL_PATH="$HOME/.config/mxmftools/orbitals/SG15-Version1p0__StandardOrbitals-Version2p0"

            if [ ! -d ".venv" ]; then
              uv venv .venv
            fi
            source .venv/bin/activate
            echo "uv pip env ready"

          '';
        };
      }
    );
}
