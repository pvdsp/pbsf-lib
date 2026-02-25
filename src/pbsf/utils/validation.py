"""Validation helpers for checking required properties and types."""

from typing import Any


def has_required(
    properties: dict[str, Any],
    required: list[tuple[str, type | list]],
) -> None:
    """
    Validate that required properties exist and have correct types.

    Parameters
    ----------
    properties : dict[str, Any]
        Dictionary of properties to validate.
    required : list[tuple[str, type | list]]
        List of (property_name, expected_type) tuples. If expected_type is a list,
        the property value must be one of the list elements.

    Raises
    ------
    ValueError
        If a required property is missing or has an incorrect type.
    """
    for (req, req_type) in required:
        if req not in properties:
            raise ValueError(f"Required property {req} not found in properties.")
        if isinstance(req_type, list):
            if properties[req] not in req_type:
                raise ValueError(
                    f"Property {req} should be one of"
                    f" {req_type}, got {properties[req]} instead."
                )
        elif not isinstance(properties[req], req_type):
            raise ValueError(
                f"Property {req} should be of type {req_type},"
                f" got {type(properties[req])} instead."
            )
