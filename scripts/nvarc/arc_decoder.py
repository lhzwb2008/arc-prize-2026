import os
import bz2
import pickle
import numpy as np

def hashable(guess):
    return tuple(map(tuple, guess))

def score_sum(guesses, getter):
    guess_list = list(guesses.values())
    scores = {}
    for g in guess_list:
        h = hashable(g["solution"])
        x = scores[h] = scores.get(h, [[], g["solution"]])
        x[0].append(g)
    scores = [(getter(sc), o) for sc, o in scores.values()]
    scores = sorted(scores, key=(lambda x: x[0]), reverse=True)
    ordered_outputs = [x[-1] for x in scores]
    return ordered_outputs

def getter_full_probmul_3(guesses, baseline=3):
    inf_score = np.sum([baseline-g["beam_score"] for g in guesses])
    aug_score = np.mean([np.sum([baseline-s for s in g["score_aug"]]) for g in guesses])
    return inf_score + aug_score

def score_full_probmul_3(guesses):
    return score_sum(guesses, getter_full_probmul_3)

def getter_kgmon(guesses):
    inf_score = len(guesses)
    aug_score = np.mean([np.mean(g["score_aug"]) for g in guesses])
    return inf_score - aug_score

def score_kgmon(guesses):
    return score_sum(guesses, getter_kgmon)


def getter_mean_quality(guesses):
    # Average sample quality; no vote-count term.
    # Only safe inside one TTT adapter. Mixing two LoRA runs' beam scores
    # lets a singleton high-q wrong grid beat a well-supported gold.
    return float(np.mean([-g["beam_score"] - np.mean(g["score_aug"]) for g in guesses]))


def score_mean_quality(guesses):
    return score_sum(guesses, getter_mean_quality)


def merge_keep_primary(sel_a, sel_p):
    """Pooled ranking, but pass-A top-1 is always one of the two attempts."""
    selected = {}
    n_forced = 0
    for bk in set(sel_a) | set(sel_p):
        a1 = (sel_a.get(bk) or [None])[0]
        top = list(sel_p.get(bk) or [])[:2]
        if a1 is not None and not any(np.array_equal(a1, g) for g in top):
            top = (top[:1] + [a1]) if top else [a1]
            n_forced += 1
        selected[bk] = top
    print(f"keep-primary: forced pass-A top-1 back on {n_forced} outputs", flush=True)
    return selected


def merge_pass_pair(sel_a, sel_b):
    """Rank each pass alone, then attempt_1=A top-1, attempt_2=B top-1.

    Tasks with no B pickle keep A's top-2 (leftover-B floor). B top-1 that
    matches A top-1 falls through to A's #2. Never mixes uncalibrated
    beam scores from two adapters into one mean_quality list.
    """
    selected = {}
    n_a_only = n_pair = n_same = n_b_only = 0
    for bk in set(sel_a) | set(sel_b):
        a = list(sel_a.get(bk) or [])
        b = list(sel_b.get(bk) or [])
        out = []
        used_b = False
        if a:
            out.append(a[0])
        if b:
            if not out:
                out.append(b[0])
                used_b = True
            elif not np.array_equal(b[0], out[0]):
                out.append(b[0])
                used_b = True
            else:
                n_same += 1
        if len(out) < 2:
            rest = (a[1:] if a else []) + (b[1:] if b else [])
            for g in rest:
                if not any(np.array_equal(g, x) for x in out):
                    out.append(g)
                    break
        selected[bk] = out[:2]
        if not b:
            n_a_only += 1
        elif not a:
            n_b_only += 1
        elif used_b:
            n_pair += 1
    print(
        f"pass-pair: {len(selected)} outputs  A-only={n_a_only} "
        f"A1+B1={n_pair} B1==A1={n_same} B-only={n_b_only}",
        flush=True,
    )
    return selected


selection_algorithms = [
    score_full_probmul_3,
    score_kgmon,
    score_mean_quality,
]


class ArcDecoder:
    
    def __init__(self, dataset, n_guesses):
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.decoded_results = {}

    def load_decoded_results(self, store, run_name=""):
        if not store or not os.path.isdir(store):
            return
        for key in os.listdir(store):
            path = os.path.join(store, key)
            if not os.path.isfile(path):
                continue
            try:
                with bz2.BZ2File(path) as f:
                    outputs = pickle.load(f)
            except Exception as e:
                print(f"skip pickle {key}: {e}", flush=True)
                continue
            base_key = key.split(".")[0]
            self.decoded_results[base_key] = self.decoded_results.get(base_key, {})
            for i, sample in enumerate(outputs):
                self.decoded_results[base_key][f"{key}{run_name}.out{i}"] = sample

    def run_selection_algo(self, selection_algorithm=score_mean_quality):
        return {bk: selection_algorithm({k: g for k, g in v.items()}) for bk, v in self.decoded_results.items()}

    def benchmark_selection_algos(self):
        print("*** Benchmark selection algorithms...")

        labels = {}
        num_tasks_per_puzzle = {}
        num_solved_keys = 0
        num_total_keys = 0

        correct_beam_scores = []

        for basekey, basevalues in self.decoded_results.items():

            mult_key, mult_sub = basekey.split("_")
            num_tasks_per_puzzle[mult_key] = max(num_tasks_per_puzzle.get(mult_key, 0), int(mult_sub) + 1)

            labels[basekey] = correct_solution = self.dataset.replies[basekey][0]

            for subkey, sample in basevalues.items():

                solution = sample["solution"]
                beam_score = sample["beam_score"]
                aug_mean = np.mean(sample["score_aug"])

                if np.shape(correct_solution) != np.shape(solution):
                    corr_str = "bad_xy_size"
                elif np.array_equal(correct_solution, solution):
                    corr_str = "ALL_CORRECT"
                    num_solved_keys += 1
                    correct_beam_scores.append(beam_score)
                else:
                    corr_str = "bad_content"

                output_len = f"{solution.shape[0]}x{solution.shape[1]}"

                if corr_str == "ALL_CORRECT":
                    print(f"{corr_str}:{beam_score:8.5f} - {aug_mean:8.5f} {output_len:5s} [{subkey}]")
                num_total_keys += 1

        print(f" subkeys: {num_solved_keys}/{num_total_keys}")
        print(f" avg correct beam score: {np.mean(correct_beam_scores):8.5f}")
        print(f" max correct beam score: {np.max(correct_beam_scores):8.5f}")

        num_puzzles = len(num_tasks_per_puzzle)

        for selection_algorithm in selection_algorithms:
            name = selection_algorithm.__name__
            selected = self.run_selection_algo(selection_algorithm)
            correct_puzzles = {k for k, v in selected.items() if any(np.array_equal(guess, labels[k]) for guess in v[:self.n_guesses])}
            print(correct_puzzles)
            score = sum(1/num_tasks_per_puzzle[k.split("_")[0]] for k in correct_puzzles)
            print(f" acc: {score:5.1f}/{num_puzzles:3} ('{name}')")