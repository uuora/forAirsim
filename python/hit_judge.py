"""Collision evidence classifier; distance alone never confirms a hit."""


def classify_collision(collision, previous_timestamp, target_name):
    if not collision.has_collided or collision.time_stamp <= previous_timestamp:
        return "NONE"
    return "TARGET" if collision.object_name == target_name else "OTHER"
