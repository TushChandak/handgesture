---
name: godot-desktop-pet
description: Use when building, adapting, or reviewing a Godot desktop-pet project, especially work inspired by Godot Asset Library asset 1294 (Desktop Pet).
---

# Godot Desktop Pet

## Use this skill when

- The user wants to create or extend a desktop-pet project in Godot.
- The user references Godot Asset Library asset 1294, "Desktop Pet".
- The task involves a Godot addon, editor plugin, or gameplay/system architecture for a desktop companion app.

## Working assumptions

- Treat asset 1294 as inspiration and reference material.
- Preserve MIT attribution when reusing upstream code or assets.
- Prefer small, composable scenes and scripts over a single monolithic controller.

## Workflow

1. Inspect the project layout first.
   - Look for `project.godot`, `addons/`, `plugin.cfg`, autoloads, and platform-specific window settings.
2. Identify the target integration type.
   - If the user wants an editor extension, scaffold under `addons/<plugin_name>/` with `plugin.cfg` and `plugin.gd`.
   - If the user wants an in-game or desktop runtime pet, prefer scenes plus one narrow autoload for global state.
3. Separate features by responsibility.
   - Keep pet behavior, UI panels, shortcuts, and minigames in distinct scripts or scenes.
   - Avoid coupling desktop-window control logic directly to minigame or note-taking logic.
4. Review desktop-specific constraints before editing.
   - Check always-on-top behavior, transparent window support, drag handling, input passthrough, and platform differences for Windows, macOS, and Linux.
5. Keep changes easy to test.
   - Add a minimal scene or plugin entry point first.
   - Then layer optional features like notes, calculators, trays, or minigames.

## Output guidance

- Prefer concrete Godot file layouts and script stubs over abstract advice.
- Call out engine-version assumptions when they affect APIs.
- If the project is not yet a Godot project, scaffold the smallest viable structure first and state what is still missing.
