let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.pandas
      python-pkgs.requests
      python-pkgs.numpy
      python-pkgs.filterpy
      python-pkgs.jupyterlab
      python-pkgs.notebook
      python-pkgs.matplotlib
      python-pkgs.seaborn
      python-pkgs.openpyxl
    ]))
  ];
}

