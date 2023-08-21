# Mojo-muse Devguide

## Architechture

`Mojo-muse` is eager to follow SOLID principles to ease the maintainance burden.
There is no circular imports in `mojo_muse`.

The dependency graph is like this:

Individual modules < models < project < resolver

Do not mess it.

The graph detail is like:

link < exceptions, utils, termui < session, auth, models/link < finders < models/specifiers, backends < markers, toml_file, config < project_file, requirements < setup, candidates, repositories < project < resolever

## About ABC

Avoid writing language specific things in abc, and the platform groundwork cpython doesn't support directly.