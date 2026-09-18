# E2 result bundle

This directory contains the closed, offline E2 comparison of the immutable
UNI 14x14 margin projected to 256x256 by bilinear interpolation, color guided
filtering, and joint bilateral upsampling. All three methods use the same
standard CPS checkpoints, validation images, GT masks, and CPS state masks.

Fixed parameters: guided radius `8`, guided epsilon `1e-3`; JBU token radius
`2`, spatial sigma `32` target pixels, range sigma `0.12` normalized RGB.
RGB is used only as propagation guidance; GT is never used in propagation.
