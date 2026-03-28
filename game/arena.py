"""Arena definitions and spacing zone classification."""

from game.combat.actions import SpacingZone


def classify_spacing(distance_sub: int, close_max: int, mid_max: int,
                     sub_pixel_scale: int) -> SpacingZone:
    """Classify the distance between fighters into a spacing zone.

    Args:
        distance_sub: Absolute distance in sub-pixel units.
        close_max: Close zone upper bound in pixels.
        mid_max: Mid zone upper bound in pixels.
        sub_pixel_scale: Sub-pixel multiplier.
    """
    distance_px = distance_sub / sub_pixel_scale
    if distance_px <= close_max:
        return SpacingZone.CLOSE
    if distance_px <= mid_max:
        return SpacingZone.MID
    return SpacingZone.FAR
