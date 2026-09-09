from __future__ import annotations
import matplotlib.colors as mcolors
from difflib import get_close_matches
import numpy as np

class MatplotlibColorError(Exception):
    pass

def suggest_color(color_name):
    """
    Suggests similar color names based on the given color name.

    Parameters
    ----------
    color_name (str):
        The name of the color for which similar color names are suggested.

    Returns
    -------
    list:
        A list of similar color names.
    """
    # Implementation of the function goes here
    all_colors = list(mcolors.CSS4_COLORS.keys())
    suggestions = get_close_matches(color_name, all_colors, n=3, cutoff=0.6)
    return suggestions

def from_matplotlib(color_name: str | int, alpha=1.0):
    """
    Converts a color name or index from Matplotlib to a hexadecimal representation.

    Parameters
    ----------
    color_name (str | int):
        The color name or index to convert.
    alpha (float, optional):
        The alpha value of the color. Defaults to 1.0.

    Returns
    -------
    int:
        The hexadecimal representation of the color in the format 0xAARRGGBB.
    """
    try:
        # Convert the color name to RGB
        rgb = mcolors.to_rgb(color_name)

        # Convert RGB (0-1 scale) to 0-255 scale and include full alpha
        r, g, b = [int(x * 255) for x in rgb]
        a = int((1-alpha) * 255)

        # Format as 0xAARRGGBB
        return int(f"0x{a:02X}{r:02X}{g:02X}{b:02X}", 16)
    except ValueError as e:
        suggestions = suggest_color(color_name)
        if suggestions:
            raise MatplotlibColorError(
                f"Invalid color name: {color_name}. Did you mean {suggestions}?"
            ) from e
        else:
            raise MatplotlibColorError(f"Invalid color name: {color_name}.") from e

def format_color(color: str | int, alpha=1.0):
    """
    Formats the given color value.

    Parameters
    ----------
    color (str | int):
        The color value to be formatted. It can be either a string representing a color
        name or a hexadecimal color code (e.g., "#FF0000" or "0xFF0000") or an integer 
        representing a color value.
    alpha (float, optional):
        The alpha value (opacity) of the color. Defaults to 1.0.

    Returns
    -------
    int:
        The formatted color value as an integer.
    """
    if isinstance(color, int):
        return color
    if color.startswith("0x"):
        return int(color, 16)
    return from_matplotlib(color, alpha)

def wvl2rgb(wvl):
    """
    Converts a given wavelength to an RGB color value.

    Parameters
    ----------
    wvl (float):
        The wavelength value in nanometers.

    Returns
    -------
    numpy.ndarray:
        An array representing the RGB color value.
    """

    if wvl < 380 or wvl > 780:
        raise ValueError("Wavelength must be between 380 and 780 nm")
    if wvl < 440:
        r = -(wvl - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif wvl < 490:
        r = 0.0
        g = (wvl - 440) / (490 - 440)
        b = 1.0
    elif wvl < 510:
        r = 0.0
        g = 1.0
        b = -(wvl - 510) / (510 - 490)
    elif wvl < 580:
        r = (wvl - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif wvl < 645:
        r = 1.0
        g = -(wvl - 645) / (645 - 580)
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0
    return np.array([r, g, b])

# Example usage
if __name__ == "__main__":    

    print("Matplotlib colors converted to 0xAARRGGBB format:")
    for color in ["red", "green", "blue",
                  "0xFFFF00FF"]:
        hex_color = format_color(color)

        if hex_color:
            as_hex = "0x" + hex(hex_color)[2:].zfill(8).upper()
            print(f"{color}: {as_hex}")
        else:
            print(f"{color}: Invalid color name")

    # Test with user input
    while True:
        user_color = input(
            "\nEnter a matplotlib color name (or 'quit' to exit): "
        ).strip()
        if user_color.lower() == "quit":
            break
        hex_color = format_color(user_color)
        
        if hex_color:
            as_hex = '0x'+hex(hex_color)[2:].zfill(8).upper()
            print(f"{user_color}: {as_hex}")
        else:
            print(f"{user_color}: Invalid color name")

