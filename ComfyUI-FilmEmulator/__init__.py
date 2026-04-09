from .film_emulator_node import AnalogFilmEmulator

NODE_CLASS_MAPPINGS = {
    "AnalogFilmEmulator": AnalogFilmEmulator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnalogFilmEmulator": "Analog Film Emulation"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']