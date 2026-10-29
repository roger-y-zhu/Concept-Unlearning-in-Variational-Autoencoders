import datetime
import subprocess


def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def fmt_12h(dt: datetime) -> str:
    return dt.strftime("%I:%M:%S %p")



def speak(text: str) -> None:
    """Play text through the system speech synthesiser (Windows)."""
    subprocess.run(
        ["powershell", "-Command",
         f'Add-Type -AssemblyName System.Speech; '
         f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'],
        check=True,
    )
    print(f"Said: {text}")
