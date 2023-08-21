# mojo_muse

![](mojo_muse_logo.png)

### A packaging solution for humans.


From python to mojo. From slick CLI to friendly API.

This is the right place to fulfill your packaging dream.

It is inspired from [PDM](https://github.com/pdm-project/pdm). It is a good start.
### Installation & Usage

Currently only one command can use, `muse init`.
This command will ask you some questions and make a project skeleton for you.

- `pip install pdm`
- `pdm install -d`
- Start venv
  - Windows: `.venv\Script\activate`
  - Linux: `.venv/bin/activate`
- `cd tests`
- `mkdir t`
- `cd t`
- `muse init`

### Following Standards:

- PEP 508: Environment Markers.

### Dependency Graph

`pyreverse -o png  -ASmy src/mojo_muse` 