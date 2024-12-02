from math import log2
from numba_pokemon_prngs.xorshift import Xoroshiro128PlusRejection
from matrix_utility import *

if __name__ == "__main__":
    min_advance = int(input("Enter minimum advance (400): ") or "400")
    max_advance = int(input("Enter maximum advance (527): ") or "527")

    full_range = max_advance - min_advance + 1 + 128
    rng = Xoroshiro128PlusRejection(0)
    rng.advance(min_advance)
    const_observations = np.zeros(full_range, np.uint8)
    for i in range(full_range):
        const_observations[i] = rng.next_rand(2)
    all_seed_to_observations = np.zeros((64, full_range), np.uint8)
    for bit in range(64):
        rng.re_init(np.uint64(1 << bit), 0)
        rng.advance(min_advance)
        for i in range(full_range):
            all_seed_to_observations[bit, i] = rng.next_rand(2)

    while True:
        observations = input("Enter 64-128 observations (0/1/Empty to quit): ")
        if not observations:
            break
        if len(observations) < 64:
            print("< 64 observations")
            continue
        if len(observations) > 128:
            print("> 128 observations")
            continue
        observations = np.array(tuple(map(int, observations)), np.uint8)
        results = []
        for advance in range(min_advance, max_advance + 1):
            offset = advance - min_advance
            seed_to_observations = all_seed_to_observations[
                :, offset : offset + len(observations)
            ]
            observations_to_seed, nullbasis = generalized_inverse(seed_to_observations)
            principle = (
                (observations ^ const_observations[offset : offset + len(observations)])
                @ observations_to_seed
            ) & 1
            for key in range(1 << len(nullbasis)):
                candidate = (
                    principle ^ (int_to_bit_vector(key, len(nullbasis)) @ nullbasis) & 1
                )
                initial_seed = bit_vector_to_int(candidate)
                rng.re_init(np.uint64(initial_seed))
                rng.advance(advance)
                current_state = np.copy(rng.state)
                if all(rng.next_rand(2) == observation for observation in observations):
                    results.append((initial_seed, advance, current_state))
        seed_count = len(results)
        print(
            f"{seed_count} possible initial seeds, record ~{log2(seed_count)} more motions"
        )
        should_show = input("Show results? (y/n): ").lower() == "y"
        if should_show:
            for seed, advance, state in results:
                rng.re_init(state[0], state[1])
                rng.advance(len(observations))
                print(
                    f"Initial Seed: {seed:016X}, Starting Advance: {advance}, Current State: {rng.state[0]:016X} {rng.state[1]:016X}"
                )
