# FractalSync — Visualizer Domain Contract

> Confirmed 2026-08-24 with the Domain Expert. This document is the design authority
> for the visualizer's behavior. Implementation must serve this contract, not replace it.
> When code and contract disagree, the contract wins until the Domain Expert amends it.

## Purpose

FractalSync is a live music visualizer that performs as a **fifth member of the band**.
It renders morphing Julia sets driven by a learned model ("the Player") that listens to
the music in real time and steers the visual world in response. The intended experience
for the audience: a sober acid trip — the song's gloom lands as visual gloom, the song's
transcendence lands as euphoria and weightlessness. Built for live performance by a
Tool tribute band.

## Glossary

One concept, one name. These terms are fixed; use them identically in code, docs, and
discussion.

| Term | Meaning |
|---|---|
| **The Map** | The Mandelbrot set — the atlas of every possible Julia set |
| **c** | The current position on The Map; selects which Julia set fills the screen |
| **Inside** | c within The Map → the Julia set is connected |
| **Outside** | c beyond The Map's edge → the Julia set is disconnected dust |
| **The Shore** | The infinitely detailed edge of The Map; the richest visual structure |
| **Section** | A stretch of song with consistent character |
| **Transition** | The moment one Section hands off to another; listeners sense it 1–2 s early |
| **Energy** | Loudness of the music |
| **Brightness** | High-frequency wash (cymbals, high guitar) vs low-frequency dominance (floor toms, low Drop D) |
| **Momentum** | c's built-up velocity; the product of sustained acceleration; slow to build, slow to stop |
| **Acceleration** | Driven by Energy; directionless — a magnitude, not a heading |
| **Jitter** | Fast, small, note-level displacement layered on top of Momentum; expresses individual notes without disturbing inertia |
| **Player** | The model; has super-human reaction time and a control surface unconstrained by human hardware |
| **Controls** | The inputs a human could plausibly operate in real time (throttle, steering, taps) |
| **Physics** | The world's default behavior: friction, attractors, Momentum carry |
| **Hands off** | Silence; the Player stops pressing, Physics runs its defaults |

## The world (facts)

- The Map is the atlas of every possible Julia set. Selecting c selects the Julia set.
- Inside The Map, the Julia set is **connected**. Outside, it is **disconnected dust**.
- The Shore holds the richest structure and is infinite — the visual wealth lives there.
- Physics runs always, whether or not the Player acts:
  - Friction constantly opposes c's motion.
  - If Inside, c settles toward its lobe's center.
  - If Outside, c drifts into an orbit around |c|≈2.
- The music has two axes: **Energy** (loudness) and **Brightness** (spectral character).
- Songs are built from Sections; a Transition is the handoff between them, and
  listeners feel it coming a second or two before it lands.

## Obligations (the visualizer shall…)

1. **Perform live** off the band's sound, amplifying the song's emotion: gloom lands as
   gloom, transcendence lands as euphoria and weightlessness.
2. **Brightness governs realm**: bright music → c Outside (disconnected dust); dark
   music → c Inside (connected).
3. **Energy governs distance from The Shore**: loud → c near The Shore; quiet → c far
   from it (toward lobe centers or |c|≈2).
4. **Energy drives Acceleration**, directionless. Sustained builds accumulate Momentum;
   Momentum carries c across The Map and decays only through friction.
5. **Jitter expresses individual notes** as fast, small displacements layered on top of
   Momentum, without disturbing it.
6. **Hands off means Physics only.** There is no silence threshold — the handoff is
   continuous. In silence, c coasts, decelerates through friction, and settles into its
   region's attractor.
7. **The Player anticipates Transitions** — reading the song ahead — and steers c
   across The Shore during the final second or two, so the connected↔disconnected flip
   lands exactly on the musical Transition. The flip shall be sudden and jarring.
8. **The Player is super-human**: reaction time beyond human, control surface
   unconstrained by human hardware, and knowledge of the band's style, having studied
   live and studio recordings of the set list.
9. **Signal loss is hands off.** The world recovers within seconds of the signal
   returning, matching the audience's own recovery.

## Design discipline

The "PlayStation controller" test: every Control must be plausible for a real-time
operator — if a human couldn't work it, the model can't learn it. The Player itself is
super-human, but the *control surface* stays human-plausible.

## Open technical decisions (deliberately black-boxed here)

These were set aside during the domain session and need full technical treatment:

- How the Player learns **anticipation** (reading the song ahead)
- How Brightness gates the realm (Inside/Outside) mechanically
- The exact control surface exposed to the Player
- The reward/loss design that encodes this contract for training
