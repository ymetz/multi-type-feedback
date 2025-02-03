"""Utility function for the RLHF Blender project."""

from typing import Literal, Union

from rlhf.datatypes import FeedbackType


# Common functions
def get_reward_model_name(
    model_id: str,
    feedback_type: FeedbackType,
    postfix: int | str,
):
    """Return the name of the trained reward model by the number postfix."""
    return "_".join(
        [
            model_id,
            feedback_type,
            str(postfix),
        ]
    )
