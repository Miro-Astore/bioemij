import jax.numpy as jnp
from jaxtyping import Array, Int


def read_rna_pair_string(dot_bracket: str) -> Int[Array, "n_pairs 2"]:
    """Convert extended dot-bracket notation to base pair indices.

    Supported hierarchy:
      Open:  ( [ { < A B C D E etc.
      Close: etc. e d c b a > } ] )

    Args:
        dot_bracket: Dot-bracket notation string.

    Returns:
        jnp.ndarray: Array of shape (n_pairs, 2) containing indices of base pairs.
    """

    open_to_close = {
        "(": ")",
        "[": "]",
        "{": "}",
        "<": ">",
        "A": "a",
        "B": "b",
        "C": "c",
        "D": "d",
        "E": "e",
        "F": "f",
        "G": "g",
        "H": "h",
        "I": "i",
        "J": "j",
        "K": "k",
        "L": "l",
        "M": "m",
        "N": "n",
        "O": "o",
        "P": "p",
        "Q": "q",
        "R": "r",
        "S": "s",
        "T": "t",
        "U": "u",
        "V": "v",
        "W": "w",
        "X": "x",
        "Y": "y",
        "Z": "z",
    }

    close_to_open = {cl: op for op, cl in open_to_close.items()}

    stacks = {op: [] for op in open_to_close}
    pairs: list[tuple[int, int]] = []

    for i, ch in enumerate(dot_bracket):
        if ch == ".":
            continue

        if ch in open_to_close:
            stacks[ch].append(i)
        elif ch in close_to_open:
            op = close_to_open[ch]
            if not stacks[op]:
                raise ValueError(f"Unmatched closing '{ch}' at position {i}")
            start = stacks[op].pop()
            pairs.append((start, i))
        elif ch.isspace():
            continue
        else:
            raise ValueError(f"Invalid character '{ch}' at position {i}")

    leftovers = {op: stk for op, stk in stacks.items() if stk}
    if leftovers:
        raise ValueError(f"Unmatched opening bracket(s): {leftovers}")

    return jnp.array(pairs)
