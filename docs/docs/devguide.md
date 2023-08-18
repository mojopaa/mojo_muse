# Mojo-muse Devguide

## Architechture

`Mojo-muse` is eager to follow SOLID principles to ease the maintainance burden.
There is no circular imports in `mojo_muse`.

The dependency graph is like this:

Individual modules < models < project < resolver

Do not mess it.