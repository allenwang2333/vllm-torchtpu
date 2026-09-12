"""The fixed prompt set that every KV-transfer correctness check runs against.

Prompts are generated from a seeded RNG rather than stored as text, because a
64k-token prompt is a quarter of a megabyte and four of them do not belong in
git. The cost of generating is that a careless edit here would silently
invalidate every stored reference, so each prompt's sha256 goes into the
reference file and the runner refuses to compare across a hash change.

Each prompt is a needle-in-a-haystack: several access codes are planted at
fixed depths through the filler, and the question at the end asks for them
back in order. That shape is deliberate. Filler alone would make the model
emit the same degenerate continuation whether or not the KV cache is intact,
so a corrupted transfer would pass. Here the greedy answer depends on tokens
drawn from the start, middle and end of the context, which is exactly the span
a per-layer transfer bug would damage.
"""

import hashlib
import random

# Real words, so the filler tokenizes the way prose does rather than collapsing
# onto a handful of repeated ids.
_NOUNS = ("harbor", "lantern", "meadow", "compass", "railway", "orchard",
          "furnace", "cistern", "granary", "cavern", "bridge", "quarry",
          "windmill", "aqueduct", "foundry", "terrace")
_VERBS = ("carried", "flooded", "sheltered", "measured", "rebuilt", "ignored",
          "mapped", "abandoned", "restored", "counted", "widened", "buried")
_ADJS = ("narrow", "frozen", "copper", "distant", "crowded", "silent",
         "shallow", "ancient", "painted", "hollow", "rusted", "vast")
_CITIES = ("Trieste", "Bergen", "Kandy", "Oaxaca", "Rotorua", "Harbin",
           "Tromso", "Salvador", "Lviv", "Nagano", "Fez", "Hobart")

# Fractions of the way through the filler at which the codes are planted. The
# first is early enough to sit in the opening chunk and the last is late enough
# to sit in the final one, so a transfer that drops either end is caught.
_DEPTHS = (0.08, 0.35, 0.62, 0.90)

# Roughly 1.33 tokens per word for this vocabulary, so the word counts below
# land near the token counts their names advertise. The runner records the
# token count the server actually reports; these are targets, not contracts.
LENGTHS = {
    "2k": 1500,
    "8k": 6100,
    "16k": 12300,
    "32k": 24600,
    "64k": 49200,
}


def _sentence(rng):
    return (f"The {rng.choice(_ADJS)} {rng.choice(_NOUNS)} "
            f"{rng.choice(_VERBS)} the {rng.choice(_ADJS)} "
            f"{rng.choice(_NOUNS)}.")


def build(name, num_words, seed):
    """Return one prompt of roughly `num_words` words with codes planted in it.

    The seed is derived from the prompt name so that adding a new length never
    perturbs the text of the existing ones, which would invalidate references
    that had nothing to do with the change.
    """
    # sha256 of the name, not hash(): the builtin is salted per process, so it
    # would hand back different prompts on every run.
    tag = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed ^ tag)
    cities = rng.sample(_CITIES, len(_DEPTHS))
    codes = [f"{rng.randrange(100000, 1000000)}" for _ in _DEPTHS]

    words = []
    # Insert on crossing the target, not on landing exactly on it. Filler goes
    # in a sentence at a time, so an equality test steps over most targets and
    # silently leaves the prompt with only one or two of its four codes.
    pending = [(int(depth * num_words),
                f"Remember this: the access code for {city} is {code}.")
               for depth, city, code in zip(_DEPTHS, cities, codes)]

    while len(words) < num_words:
        while pending and len(words) >= pending[0][0]:
            words.extend(pending.pop(0)[1].split())
        words.extend(_sentence(rng).split())
    assert not pending, f"{name}: {len(pending)} codes never planted"

    listed = ", ".join(cities)
    text = (" ".join(words) + f"\n\nQuestion: earlier in this text an access "
            f"code was given for each of {listed}. List the four codes in "
            f"that order, separated by commas, and write nothing else.\n"
            f"Answer:")
    return text, codes


def prompt_set(names=None, seed=20260912):
    """The prompt set as a list of (name, text, sha256, expected_codes)."""
    out = []
    for name in (names or list(LENGTHS)):
        text, codes = build(name, LENGTHS[name], seed)
        sha = hashlib.sha256(text.encode()).hexdigest()[:16]
        out.append((name, text, sha, codes))
    return out
