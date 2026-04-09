<img width="1531" height="635" alt="image" src="https://github.com/user-attachments/assets/53ba4f02-4d6e-4014-b881-71a9a21697d0" />


# 🎞️ Procedural Analog Film Emulator (Hybrid Engine)

A professional-grade imaging pipeline designed to transform sterile, microscopically sharp digital and AI-generated images into authentic analog photographs. Available as a standalone Asynchronous UI app and a ComfyUI Custom Node.

# 🧠 The Philosophy: Why Not Just Use LUTs?

Most "film filters" rely on 3D LUTs (.cube files). A LUT is a rigid mathematical grid that shifts colors; it has no concept of physical space, light scatter, or emulsion thickness. It feels "pasted on" because it is.

# 🔬 This engine utilizes a Hybrid Procedural Pipeline, inspired by high-end cinema grading too

The Empirical Core: It uses JSON-based characteristic curves extracted from real film stocks to perfectly map exposure densities.
The Procedural Physics: It wraps those curves in real-time physical simulations of vintage glass, chemical light scatter, darkroom printing, and signal-dependent silver halide grain.

# 🎛️ Parameter Guide (What the Sliders Do)

1. Lens Physics (Pre-Exposure)
AI generators (like Midjourney/Stable Diffusion) and modern digital sensors create mathematically perfect, infinitely sharp edges. Vintage lenses do not.

* Optical Softness: Applies a microscopic defocus. It breaks the "plastic" AI look by softening edge contrast before the light hits the virtual film.

* Chromatic Aberration: Simulates imperfect lens refraction by laterally shifting the Red channel outward and the Blue channel inward, creating subtle color fringing on the edges of objects.

2. Light & Emulsion Exposure
How light behaves physically inside the camera body and the film emulsion.

* Cine-Log Flattening: Pre-flattens harsh digital contrast, mimicking the flat, data-rich look of a raw Cineon/Log film scan.

* Dual-Stage Halation: True film halation is a chemical reflection off the back of the film base that bleeds back into the Red emulsion layer. This slider extracts intense highlights and creates a localized, fiery red/orange glow around high-contrast edges.

* Lens Bloom: Unlike halation, bloom is optical. It simulates light scattering broadly inside the glass of a vintage lens, creating a wide, soft white/cyan glow around bright areas.

3. Darkroom Print & Chemistry
What happens when the developed negative is projected onto photographic paper.

* Analog Print Contrast: Applies a physical optical S-Curve. It compresses the highlights into a smooth roll-off and crushes the shadows, giving the image punch and density.
* Warm/Cool Split Tone: Simulates classic darkroom chemistry imbalances. Pushing this positive warms the midtones (yellow/orange) while cooling the shadows (cyan/blue). Pushing it negative creates a gritty, matrix-style reversal.

4. Physical Grain & Output
Digital noise is uniform TV static. Analog grain is physical dye clouds that clump together and react to exposure levels.

* Dye Cloud Grain Amount: Controls the intensity of the grain. This uses True Luma Masking—grain peaks aggressively in the shadows and midtones, but is bleached away in pure white highlights, exactly like real silver halide.

* Crystal Size / Coarseness: Simulates the physical size of the ISO crystals. Higher values create larger, softer clumps of grain (e.g., mimicking an 800-speed film vs a 100-speed film).

* Overall Physical Mix: A global opacity slider to blend the final analog simulation back over your original image.

# 🚀 Installation & Usage

Install as a usual Git.
Film Profiles are included in \profiles folder (every emulsion preset has 2 corresponding files: JSON & CSV)

Required packages (pretty much standard set included with python env):

numpy
cv2,
json,
csv,
torch,
scipy.ndimage

