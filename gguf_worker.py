from PyQt6.QtCore import (
    QThread, pyqtSignal
)
from gguf.gguf_reader import GGUFReader
import numpy as np
from pathlib import Path

class GGUFInfoWoker(QThread):
    info_ready = pyqtSignal(dict)

    def __init__(self, model_path):
        super().__init__()
        self._is_running = True
        self.model_path = model_path
        self.weights_map = {
            0: "F32",
            1: "F16",
            2: "Q4_0",
            3: "Q4_1",
            4: "Q4_1_SOME_F16",
            5: "Q4_2",
            6: "Q4_3",
            7: "Q8_0",
            8: "Q5_0",
            9: "Q5_1",
            10: "Q2_K",
            11: "Q3_K_S",
            12: "Q3_K_M",
            13: "Q3_K_L",
            14: "Q4_K_S",
            15: "Q4_K_M",
            16: "Q5_K_S",
            17: "Q5_K_M",
            18: "Q6_K",
            19: "IQ2_XSS",
            20: "IQ2_XS",
            21: "Q2_K_S",
            22: "IQ3_XS",
            23: "IQ3_XXS",
            24: "IQ1_S",
            25: "IQ4_NL",
            26: "IQ3_S",
            27: "IQ3_M",
            28: "IQ2_S",
            29: "IQ2_M",
            30: "IQ4_XS",
            31: "IQ1_M",
            32: "BF16",
            33: "Q4_0_4_4",
            34: "Q4_0_4_8",
            35: "Q4_0_8_8",
            36: "TQ1_0",
            37: "TQ2_0",
            38: "MXFP4_MOE",
            145: "IQ4_KS",
            147: "IQ2_KS",
            148: "IQ4_KSS",
            150: "IQ5_KS",
            154: "IQ3_KS",
            155: "IQ2_KL",
            156: "IQ1_KT"
        }

    def run(self):
        self._is_running = True
        info = {}
        try:
            model_info = GGUFReader(self.model_path)
            info.update({"path": self.model_path})
            for key, field in model_info.fields.items():
                if key == "general.name":
                    info.update({"name": str(self.maybe_decode(field.parts[field.data[0]]))})
                elif key == "general.size_label":
                    info.update({"parameters": str(self.maybe_decode(field.parts[field.data[0]]))})
                elif key == "general.file_type":
                    info.update({"weights": self.weights_map.get(self.maybe_decode(field.parts[field.data[0]]), f"Unknown ({field.data[0]})")})
                elif key.endswith("block_count"):
                    info.update({"layers": str(self.maybe_decode(field.parts[field.data[0]]))})
            path = Path(self.model_path)
            size_bytes = path.stat().st_size
            for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
                if size_bytes < 1024:
                    info.update({"size": f"{size_bytes:.2f} {unit}"})
                    break
                size_bytes /= 1024
            if "name" not in info:
                info["name"] = "Unknown"
            if "parameters" not in info:
                info["parameters"] = "Unknown"
            if "weights" not in info:
                info["weights"] = "Unknown"
            if "layers" not in info:
                info["layers"] = "Unknown"
            if "size" not in info:
                info["size"] = "Unknown"
        except Exception as e:
            info = {"path": self.model_path, "name": "Error", "parameters": "Error", "weights": "Error", "layers": "Error", "size": "Error"}

        self.info_ready.emit(info)

        if not self._is_running:
            return
        
    def stop(self):
        self._is_running = False
    
    def maybe_decode(self, value):
        # bytes or bytearray -> try UTF-8
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", errors="replace")

        # numpy array of uint8 -> bytes -> UTF-8
        if isinstance(value, np.ndarray) and value.dtype == np.uint8:
            return value.tobytes().decode("utf-8", errors="replace")

        # list/tuple of ints 0..255 -> bytes -> UTF-8
        if isinstance(value, (list, tuple)) and value and all(isinstance(x, (int, np.integer)) and 0 <= int(x) <= 255 for x in value):
            return bytes(value).decode("utf-8", errors="replace")

        # list/tuple of bytes -> decode each
        if isinstance(value, (list, tuple)) and value and all(isinstance(x, (bytes, bytearray)) for x in value):
            return [x.decode("utf-8", errors="replace") for x in value]

        return value[0]