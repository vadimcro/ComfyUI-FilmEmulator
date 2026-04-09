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

# 📚 The story Behind the project.
Working with ComfyUI, I generate hundreds of images every week. A good portion of them end up as commercial-grade visuals for advertising campaigns. I work with many AI models — Flux, Qwen, NanoBanana, and others — and each of them brings something valuable to the table.

But most of them also share the same problem.

And no, I do not mean only the famous “plastic skin” effect.

The deeper issue is that AI images are often just too perfect. Too balanced. Too clean. Too correct. No real exposure mistakes, no strange color drift, no subtle tint bias, no unexpected roll-off, no sensor character, no lens flaws, no grain, no noise, no chemistry. It feels as if a mythical camera with flawless exposure metering, dead-on white balance, superhuman color separation, surgical sharpness, and zero ISO noise produced a file from another universe.

Coming from 24 years in photography, I know that universe does not exist.

Every film emulsion is imperfect. Every sensor is imperfect. Every camera and every lens leaves it’s signature on the image. That is exactly why straight-out-of-camera files have rarely been the final destination.

Back in 2009, while I was transitioning from medium and large format film into a digital workflow, I had a long conversation over a glass of wine with Pavel Kosenko, the mind behind what many people now know as DEHANCER. Pavel had an extraordinary understanding of film — processing, scanning, printing, the whole chain. He talked about bringing a chemistry-driven analogue image language into a digital world ruled by math.

At the time, it sounded almost absurd.

Why would anyone intentionally make a “perfect” digital file worse?

Why would you dehance an image?

We went our separate ways, and life moved on. Then, about a week ago, that old conversation came back to me with new meaning.

I thought: what if I built a dehancing tool specifically for my AI workflow?

Not as a nostalgia filter. Not as another LUT pack. But as a way to bring back the beautiful imperfections that make images feel photographed rather than generated. A way to take AI outputs and reintroduce the kinds of tonal and color behavior I know from hands-on experience — Velvia RVP, Kodak Ektar, Fujifilm NPS PRO 160S, and the rest of that analogue memory bank.

And that is how this little side project started.

At first, the idea felt straightforward. Build a film-emulation engine. Feed it the behavior of real film. Apply it to AI images. Done.

Of course, it was not that simple.

The first big decision was conceptual: should the tool imitate film using a scientific route, based on datasheets and sensitometric curves, or a perceptual route, based on how film actually looks after being exposed, developed, scanned, and viewed?

The scientific path was seductive. Characteristic curves, spectral sensitivity, density response — it all looked wonderfully precise. But very quickly I realized something important: precision on paper does not automatically translate into convincing images on screen. Film datasheets describe a material under controlled conditions. They do not contain the full messiness of real-life film: the developer, the scanner, the print chain, the subtle drift in color that happens across the tonal range.

Still, those datasheets were too valuable to ignore. So instead of making them the entire solution, I made them the foundation for the first module.

That led to the next decision — and the first real dead end.

My initial thought was to build a smart extraction tool that could read film datasheets, parse the graphs automatically, and generate the data needed for the engine. On paper, that sounded elegant. In reality, it was a trap. Vintage datasheets are inconsistent, low-resolution, often scanned badly, and full of overlapping black-and-white graphs. Some curves are not color-coded at all. Some are distinguished only by dotted or dashed lines. In other words: perfect material for human interpretation, terrible material for blind automation.

That was an important turning point. Instead of forcing a clever but brittle OCR-and-computer-vision solution, I pivoted to a semi-manual extractor tool. A one-time utility designed to turn screenshots of technical graphs into clean CSV and JSON files.

That decision changed everything.

The project became much more grounded. Instead of trying to build an AI that understands 30 years of film documentation, I built a reliable workflow. Capture the graph. Calibrate it. Trace the relevant curves. Save the data. Store the film stock with metadata: manufacturer, stock name, ISO, process type, D-min values, curve files, spectral files. Not glamorous, but robust.

That extractor became the quiet backbone of the whole project.

Then came the process engine itself. The early versions were almost embarrassingly simple: push or pull exposure, run the image through film curves, blend it back with the original. It worked, but it felt too blunt. The image got heavier, denser, often muddy. So the engine started evolving.

Flattening was added before the film mapping to soften the “perfect digital” look. Halation came next, and the first version was so subtle it was practically invisible. Then it became too aggressive. Then it finally found its place. Along the way, I learned again what every image-maker eventually learns: effects are easy, believable effects are hard.

There were more wrong turns. At one point, I explored CMY crosstalk and print-contrast style logic that sounded intellectually satisfying but turned into a mess in practice. Some parameters looked great in the UI and did almost nothing in the image. That was another lesson: if a slider does not have a living, visible mathematical consequence, it should not exist. Dead controls are worse than no controls.

Slowly, the tool became more honest.

The order of operations mattered. Linear light mattered. Exposure mapping mattered. Halation had to happen in the right place. Negative and reversal films had to branch differently. Density had to be treated as transmittance instead of some abstract contrast gimmick. Every little refinement made the result feel less like a filter and more like a photographic process.

That was probably the most satisfying part of the whole build: watching the project move away from “cool effect” territory and closer to “this actually behaves like an image-making pipeline.”

And maybe that is the real success here.

Not that I built a finished product in a week. I did not.

The success is that the project found its own logic.

It stopped pretending that film look can be reduced to a LUT and a grain overlay. It stopped chasing clever automation where careful manual work was the better answer. It stopped adding controls for the sake of complexity. And it started behaving more like the thing that inspired it in the first place: a chain of imperfect, physical, beautiful compromises.

Which, if I am honest, is probably why this idea pulled me in so strongly.

AI images are often impressive. Sometimes stunning. But they are still missing the quiet human truth of imperfect capture. They need a little friction. A little chemistry. A little unpredictability. A little history.

A little dehancing.


And for the first time in years, I found myself back in the same territory Pavel was talking about over that glass of wine: trying to bring analogue behavior into a world of numbers.

Only now, the target is not digital photography.

It is AI.

And that makes the whole thing feel strangely full-circle.
