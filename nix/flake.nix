{
  description = "Python development environment with uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
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
        };
      in
      {
        devShells.default = pkgs.mkShell rec{
          buildInputs = with pkgs; [

            uv
            basedpyright
            ruff


            #for libstdc++.so.6
            stdenv.cc.cc
            # for libz.so.1
            zlib

            python3Packages.pyqt6
            qt6.full

          ];

          MPLBACKEND = "QtAgg";
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
