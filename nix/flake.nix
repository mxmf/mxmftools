{
  description = "Python development environment with uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
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
          config.allowUnfree = true;
          cudaSupport = true;
        };
        pkgs-unstable = import nixpkgs {
          system = system;
          config.allowUnfree = true;
          cudaSupport = true;
        };

      in
      {
        devShells.default = pkgs.mkShell rec{
          buildInputs = with pkgs; [
            nix

            uv
            basedpyright
            ruff

            cudatoolkit
            # cudaPackages.cudatoolkit # CUDA Toolkit für GPU-Unterstützung
            # cudaPackages.cudnn # CUDA Deep Neural Network library
            # (pkgs-unstable.linuxPackages.nvidia_x11)


            #for libstdc++.so.6
            stdenv.cc.cc
            # for libz.so.1
            zlib

            python3Packages.pyqt5
            qt5.qtwayland
            libsForQt5.qt5.qtbase


          ];


          MPLBACKEND = "QtAgg";

          QT_PLUGIN_PATH = "${pkgs.qt5.qtbase}/${pkgs.qt5.qtbase.qtPluginPrefix}";

          # LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
          CUDA_PATH = pkgs.cudatoolkit;
          NVIDIA_DRIVERS = "/run/opengl-driver/lib";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (buildInputs ++ [ "/run/opengl-driver" ]);
          EXTRA_LDFLAGS = "-L/lib -L/run/opengl-driver/lib";

          shellHook = ''

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
