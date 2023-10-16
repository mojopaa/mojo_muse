- https://peps.python.org/pep-0610/#git

- Move BaseEnvironment.get_finder() to Project bc it needs pyproject.toml's source
- Deprecate Environment class and join it to Project.
- Convert pyproject.toml to mojoproject.toml
- Rewrite the hard-coding project.pyproject to project.project_file