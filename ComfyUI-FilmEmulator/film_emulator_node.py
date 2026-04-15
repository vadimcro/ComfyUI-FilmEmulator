import numpy as np
import cv2
import json
import csv
import os
import torch

# Get the directory of this script to locate the 'profiles' folder
NODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(NODE_DIR, "profiles")

class AnalogFilmEmulator:
    @classmethod
    def INPUT_TYPES(s):
        profile_list = []
        if os.path.exists(PROFILES_DIR):
            profile_list = [f for f in os.listdir(PROFILES_DIR) if f.endswith('.json')]
        if not profile_list:
            profile_list = ["No profiles found"]

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image (AI generated or digital photo)."}),
                "film_profile": (profile_list, {"tooltip": "Select the empirical JSON characteristic curve."}),
                
                # 1. LENS OPTICS
                "optical_softness": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Simulates vintage lens defocus to break artificial digital sharpness."}),
                "chromatic_aberration": ("INT", {"default": 2, "min": 0, "max": 20, "step": 1, "tooltip": "Lateral color fringing (shifts Red and Blue channels)."}),
                
                # 2. LIGHT & SCATTER
                "cinelog_flattening": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Pre-flattens digital contrast to mimic a Cine-Log scan."}),
                "halation_amount": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Red emulsion scatter localized around high-contrast bright edges."}),
                "lens_bloom": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Wide, soft white glow simulating light scattering inside a vintage lens."}),
                
                # 3. PRINT CHEMISTRY
                "print_contrast": ("FLOAT", {"default": 40.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Applies an analog darkroom S-Curve for dense shadows and rolled highlights."}),
                "split_tone": ("FLOAT", {"default": 15.0, "min": -50.0, "max": 50.0, "step": 1.0, "tooltip": "Pushes midtones warm (yellow/red) and shadows cool (cyan/blue). Negative values reverse this."}),
                
                # 4. GRAIN & MIX
                "grain_amount": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Intensity of the physical silver-halide dye clouds. Peaks in midtones, vanishes in pure whites."}),
                "grain_size": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 10.0, "step": 0.1, "tooltip": "Physical coarseness/clumping size of the grain crystals."}),
                "overall_mix": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Opacity blend of the final analog effect over the original image."}),
                
                # 5. EDGE IMPERFECTIONS (Applied Last)
                "field_flatness_amt": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Simulates optical degradation and softness towards the extreme edges of the lens elements."}),
                "field_flatness_falloff": ("FLOAT", {"default": 40.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "0% = Only absolute extreme corners. 100% = Softness creeps heavily towards the center."}),
                "vignette_intensity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Darkens the corners of the image due to physical lens barrel shadow."}),
                "vignette_falloff": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "0% = Only absolute extreme corners. 100% = Shadow creeps heavily towards the center."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_emulation"
    CATEGORY = "image/postprocessing"

    def apply_emulation(self, image, film_profile, optical_softness, chromatic_aberration, cinelog_flattening, 
                        halation_amount, lens_bloom, print_contrast, split_tone, grain_amount, grain_size, overall_mix,
                        field_flatness_amt, field_flatness_falloff, vignette_intensity, vignette_falloff):
        
        # 1. Load JSON Profile
        profile_path = os.path.join(PROFILES_DIR, film_profile)
        try:
            with open(profile_path, 'r') as f:
                profile_data = json.load(f)
            
            csv_filename = profile_data['data_files']['characteristic_curve']
            csv_full_path = os.path.join(PROFILES_DIR, csv_filename)
            
            curve_data = {'r': [], 'g': [], 'b': []}
            log_e_grid = []
            with open(csv_full_path, 'r') as f:
                reader = csv.reader(f)
                next(reader) 
                for row in reader:
                    log_e_grid.append(float(row[0]))
                    curve_data['r'].append(float(row[1]))
                    curve_data['g'].append(float(row[2]))
                    curve_data['b'].append(float(row[3]))
            log_e_grid = np.array(log_e_grid)
            for ch in ['r', 'g', 'b']:
                curve_data[ch] = np.array(curve_data[ch])
        except Exception as e:
            print(f"Error loading profile: {e}")
            return (image,) 

        # 2. Process Image Batch
        processed_batch = []
        for i in range(image.shape[0]):
            img_np = image[i].cpu().numpy()
            
            out_np = self.process_engine(
                img_array=img_np,
                soft_amt=optical_softness,
                ca_amt=chromatic_aberration,
                flatten_pct=cinelog_flattening,
                hal_pct=halation_amount,
                bloom_pct=lens_bloom,
                contrast_pct=print_contrast,
                split_pct=split_tone,
                grain_amt=grain_amount,
                grain_size=grain_size,
                strength_pct=overall_mix,
                ff_amt=field_flatness_amt,
                ff_fall=field_flatness_falloff,
                vig_amt=vignette_intensity,
                vig_fall=vignette_falloff,
                profile_data=profile_data,
                log_e_grid=log_e_grid,
                curve_data=curve_data
            )
            processed_batch.append(torch.from_numpy(out_np))

        out_tensor = torch.stack(processed_batch)
        return (out_tensor,)

    def _generate_radial_mask(self, h, w, falloff_pct):
        """ Generates a smoothstep radial mask for Vignette and Field Flatness """
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        norm_dist = dist / max_dist
        
        safe_radius = 1.0 - (falloff_pct / 100.0)
        
        mask = (norm_dist - safe_radius) / (1.0 - safe_radius + 1e-5)
        mask = np.clip(mask, 0, 1)
        # Hermite smoothstep for organic falloff
        mask = mask * mask * (3 - 2 * mask)
        return np.stack([mask]*3, axis=-1)

    def process_engine(self, img_array, soft_amt, ca_amt, flatten_pct, hal_pct, bloom_pct, 
                       contrast_pct, split_pct, grain_amt, grain_size, strength_pct, 
                       ff_amt, ff_fall, vig_amt, vig_fall,
                       profile_data, log_e_grid, curve_data):
        
        img = np.copy(img_array)
        h, w, _ = img.shape

        # --- 1. OPTICS (Base Softness & CA) ---
        soft = soft_amt / 100.0
        ca = int(ca_amt)
        if soft > 0:
            sigma = (w * 0.001) * (soft * 2)
            img = cv2.GaussianBlur(img, (0,0), sigma)

        if ca > 0:
            r, g, b = cv2.split(img)
            M_R = np.float32([[1, 0, ca], [0, 1, ca]])
            r_shift = cv2.warpAffine(r, M_R, (w, h), borderMode=cv2.BORDER_REPLICATE)
            M_B = np.float32([[1, 0, -ca], [0, 1, -ca]])
            b_shift = cv2.warpAffine(b, M_B, (w, h), borderMode=cv2.BORDER_REPLICATE)
            img = cv2.merge([r_shift, g, b_shift])

        # --- 2. EXPOSURE PREP & SCATTER ---
        linear = np.power(img, 2.2)
        
        flat_amount = flatten_pct / 100.0
        if flat_amount > 0:
            log_base = 1.0 + (flat_amount * 10.0)
            linear = np.log(1.0 + (log_base - 1.0) * linear) / np.log(log_base)

        bloom = bloom_pct / 100.0
        if bloom > 0:
            hl_mask_wide = np.clip((linear - 0.5) / 0.5, 0, 1).astype(np.float32)
            bloom_blur = cv2.GaussianBlur(hl_mask_wide, (0,0), w * 0.02)
            linear = linear + (bloom_blur * bloom * 0.5)
        
        hal = hal_pct / 100.0
        if hal > 0:
            hl_threshold = 0.4
            highlights = np.clip((linear - hl_threshold) / (1.0 - hl_threshold), 0, 1).astype(np.float32)
            sigma_core = w * 0.005 
            sigma_wide = w * 0.02  
            
            core_blur = cv2.GaussianBlur(highlights, (0,0), sigma_core)
            wide_blur = cv2.GaussianBlur(highlights, (0,0), sigma_wide)
            halation_map = (core_blur * 0.6) + (wide_blur * 0.4)
            
            tint = np.array([5.0, 1.0, 0.0], dtype=np.float32)
            linear = linear + (halation_map * tint * hal)

        # --- 3. JSON TONE ENGINE ---
        digital_log = np.log10(np.clip(linear, 1e-5, None))
        min_log_e = np.min(log_e_grid)
        max_log_e = np.max(log_e_grid)
        curve_center = (min_log_e + max_log_e) / 2.0
        
        mapped_log_e = digital_log - np.log10(0.18) + curve_center
        
        out_r = np.interp(mapped_log_e[:, :, 0], log_e_grid, curve_data['r'])
        out_g = np.interp(mapped_log_e[:, :, 1], log_e_grid, curve_data['g'])
        out_b = np.interp(mapped_log_e[:, :, 2], log_e_grid, curve_data['b'])
        density = np.stack([out_r, out_g, out_b], axis=-1)
        
        density_scalar = 0.60 
        density = density * density_scalar

        d_min_raw = profile_data['density_anchors']['d_min']
        d_max_raw = profile_data['density_anchors']['d_max']
        
        raw_min_array = np.array([d_min_raw['r'], d_min_raw['g'], d_min_raw['b']]) * density_scalar
        raw_max_array = np.array([d_max_raw['r'], d_max_raw['g'], d_max_raw['b']]) * density_scalar
        
        d_min_array = np.minimum(raw_min_array, raw_max_array)
        d_max_array = np.maximum(raw_min_array, raw_max_array)

        transmittance = 10.0 ** (-density)
        t_max = 10.0 ** (-d_min_array)  
        t_min = 10.0 ** (-d_max_array)  
        t_norm = (transmittance - t_min) / (t_max - t_min)

        if profile_data['properties']['film_type'] == 'negative':
            out_linear = 1.0 - t_norm
        else:
            out_linear = t_norm

        out_linear = np.clip(out_linear, 0.001, 0.999)
        film_out = np.power(out_linear, 1/2.2)

        # --- 4. DARKROOM CHEMISTRY ---
        contrast = contrast_pct / 100.0
        if contrast > 0:
            s_curve = (film_out ** 2) * (3.0 - 2.0 * film_out)
            film_out = (film_out * (1.0 - contrast)) + (s_curve * contrast)

        split = split_pct / 100.0
        if abs(split) > 0:
            luma = cv2.cvtColor(film_out.astype(np.float32), cv2.COLOR_RGB2GRAY)
            luma_3d = np.stack([luma]*3, axis=-1)
            warm = np.array([1.1, 1.0, 0.9])
            cool = np.array([0.9, 1.0, 1.1])
            
            if split > 0: 
                color_map = (warm * luma_3d) + (cool * (1.0 - luma_3d))
                film_out = film_out * ((1.0 - split) + (color_map * split))
            else: 
                color_map = (cool * luma_3d) + (warm * (1.0 - luma_3d))
                film_out = film_out * ((1.0 + split) + (color_map * -split))

        film_out = np.clip(film_out, 0, 1)

        max_opacity_limit = 0.50
        opacity = (strength_pct / 100.0) * max_opacity_limit
        blended = (img_array * (1.0 - opacity)) + (film_out * opacity)

        # --- 5. SIGNAL-DEPENDENT GRAIN ---
        grain = grain_amt / 100.0
        if grain > 0:
            scale = max(1.0, float(grain_size) / 1.5)
            gh, gw = int(h / scale), int(w / scale)
            noise = np.random.normal(0, 1.0, (gh, gw, 3)).astype(np.float32)
            
            if scale > 1.0:
                noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
            
            noise = np.sign(noise) * (np.abs(noise) ** 0.8)
                
            luma_blend = cv2.cvtColor(blended.astype(np.float32), cv2.COLOR_RGB2GRAY)
            grain_mask = np.power(luma_blend, 0.5) * np.power(1.0 - luma_blend, 1.5) * 2.5
            grain_mask = np.clip(grain_mask, 0.0, 1.0)
            grain_mask_3d = np.stack([grain_mask]*3, axis=-1)

            blended = blended + (noise * grain_mask_3d * grain * 0.15)

        # --- 6. EDGE IMPERFECTIONS (APPLIED LAST) ---
        if ff_amt > 0:
            ff_mask = self._generate_radial_mask(h, w, ff_fall)
            max_sigma = (w * 0.005) * (ff_amt / 100.0) 
            blurred_img = cv2.GaussianBlur(blended, (0,0), max_sigma)
            blended = (blended * (1.0 - ff_mask)) + (blurred_img * ff_mask)

        if vig_amt > 0:
            vig_mask = self._generate_radial_mask(h, w, vig_fall)
            intensity = vig_amt / 100.0
            blended = blended * (1.0 - (vig_mask * intensity))

        return np.clip(blended, 0, 1)
