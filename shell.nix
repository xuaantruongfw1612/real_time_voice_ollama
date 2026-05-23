{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python311;
  py = python.pkgs;
in
pkgs.mkShell {
  buildInputs = [
    python
    py.pip
    py.virtualenv
    py.numpy
    py.requests
    py.python-dotenv
    py.sounddevice

    pkgs.portaudio
    pkgs.pkg-config
    pkgs.ffmpeg
  ];

  shellHook = ''
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export PYTHONUNBUFFERED=1
    export OLLAMA_URL=http://localhost:11434
    echo "NixOS dev shell ready"
  '';
}