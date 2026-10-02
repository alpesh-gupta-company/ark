"""
Dataset preparation for Colab T4 training.
Downloads and prepares:
  1. FineWeb-Edu (educational web text) for base pretraining
  2. OpenOrca + Alpaca-GPT4 (instruction data) for chat fine-tuning
  3. Chain-of-Thought reasoning data (math, logic, step-by-step)

Run this in Google Colab BEFORE training.
"""

import json
import os
import random
import sys


def prepare_pretraining_data(output_path="data/colab_pretrain.txt", target_tokens_m=500):
    """
    Download educational web text for base pretraining.
    Uses HuggingFace datasets library to stream FineWeb-Edu.
    Target: ~500M tokens of high-quality educational content.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system(f"{sys.executable} -m pip install -q datasets")
        from datasets import load_dataset

    print(f"📥 Streaming FineWeb-Edu (target: ~{target_tokens_m}M tokens)...")
    os.makedirs(os.path.dirname(output_path) or "data", exist_ok=True)

    # Approximate: 1 token ≈ 4 chars. Target chars = tokens * 4
    target_chars = target_tokens_m * 1_000_000 * 4
    total_chars = 0
    doc_count = 0

    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu-score-2",
            split="train",
            streaming=True,
        )
    except Exception:
        print("⚠️ FineWeb-Edu not available. Trying C4...")
        try:
            ds = load_dataset(
                "allenai/c4",
                "en",
                split="train",
                streaming=True,
            )
        except Exception:
            print("⚠️ C4 not available. Trying OpenWebText...")
            ds = load_dataset(
                "Skylion007/openwebtext",
                split="train",
                streaming=True,
            )

    with open(output_path, "w", encoding="utf-8") as f:
        for example in ds:
            text = example.get("text", "")
            if not text or len(text) < 100:
                continue

            f.write(text + "\n\n")
            total_chars += len(text)
            doc_count += 1

            if doc_count % 5000 == 0:
                est_tokens = total_chars // 4
                print(f"  📊 {doc_count} documents | ~{est_tokens/1e6:.1f}M tokens")

            if total_chars >= target_chars:
                break

    est_tokens = total_chars // 4
    print(f"✅ Saved {doc_count} documents (~{est_tokens/1e6:.0f}M tokens) to {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1e9:.2f} GB")
    return output_path


def _generate_arithmetic_cot(count=5000):
    """
    Generate synthetic chain-of-thought arithmetic problems.
    These teach the model to show its work step-by-step.
    """
    random.seed(42)
    examples = []

    # --- Addition / Subtraction ---
    for _ in range(count // 4):
        a, b = random.randint(1, 999), random.randint(1, 999)
        op = random.choice(["+", "-"])
        result = a + b if op == "+" else a - b
        op_word = "add" if op == "+" else "subtract"

        thinking = (
            f"<think>\n"
            f"I need to {op_word} {b} {'to' if op == '+' else 'from'} {a}.\n"
            f"{a} {op} {b} = {result}\n"
            f"</think>\n\n"
            f"The answer is {result}."
        )
        examples.append({
            "messages": [
                {"role": "user", "content": f"What is {a} {op} {b}?"},
                {"role": "assistant", "content": thinking},
            ]
        })

    # --- Multiplication ---
    for _ in range(count // 4):
        a, b = random.randint(2, 99), random.randint(2, 99)
        result = a * b
        thinking = (
            f"<think>\n"
            f"I need to multiply {a} by {b}.\n"
            f"{a} × {b} = {result}\n"
            f"</think>\n\n"
            f"{a} × {b} = {result}."
        )
        examples.append({
            "messages": [
                {"role": "user", "content": f"Calculate {a} × {b}."},
                {"role": "assistant", "content": thinking},
            ]
        })

    # --- Multi-step word problems ---
    for _ in range(count // 4):
        item = random.choice(["apples", "books", "marbles", "cookies", "pencils", "stickers"])
        name1 = random.choice(["Alice", "Bob", "Sam", "Emma", "Tom", "Mia"])
        name2 = random.choice(["Charlie", "David", "Sarah", "Lily", "Jack", "Zoe"])
        initial = random.randint(5, 50)
        gave = random.randint(1, initial - 1)
        remaining = initial - gave

        question = f"{name1} has {initial} {item}. {name1} gives {gave} {item} to {name2}. How many {item} does {name1} have now?"
        thinking = (
            f"<think>\n"
            f"Let me work through this step by step.\n"
            f"1. {name1} starts with {initial} {item}.\n"
            f"2. {name1} gives away {gave} {item} to {name2}.\n"
            f"3. So {name1} has {initial} - {gave} = {remaining} {item} left.\n"
            f"</think>\n\n"
            f"{name1} has {remaining} {item}."
        )
        examples.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": thinking},
            ]
        })

    # --- Simple logic / comparison ---
    for _ in range(count // 4):
        a, b = random.randint(1, 1000), random.randint(1, 1000)
        while a == b:
            b = random.randint(1, 1000)

        bigger = a if a > b else b
        smaller = a if a < b else b
        diff = bigger - smaller

        question = random.choice([
            f"Which is larger, {a} or {b}?",
            f"Is {a} greater than {b}?",
            f"Compare {a} and {b}.",
        ])

        if a > b:
            answer_text = f"{a} is larger than {b} by {diff}."
            comparison = f"{a} > {b}"
        else:
            answer_text = f"{b} is larger than {a} by {diff}."
            comparison = f"{a} < {b}"

        thinking = (
            f"<think>\n"
            f"I need to compare {a} and {b}.\n"
            f"{comparison}, so the difference is {diff}.\n"
            f"</think>\n\n"
            f"{answer_text}"
        )
        examples.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": thinking},
            ]
        })

    random.shuffle(examples)
    return examples


def _generate_logic_cot(count=3000):
    """
    Generate simple logical reasoning examples with chain-of-thought.
    """
    random.seed(123)
    examples = []

    animals = ["dogs", "cats", "birds", "fish", "horses", "rabbits", "snakes"]
    colors = ["red", "blue", "green", "yellow", "black", "white", "orange"]
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]

    # --- Syllogisms ---
    for _ in range(count // 3):
        category = random.choice(["mammals", "reptiles", "living things", "vehicles", "fruits"])
        member = random.choice(["a dog", "a cat", "a car", "an apple", "a snake", "a tree"])
        property_ = random.choice(["needs water", "has mass", "takes up space", "exists in nature", "can be described"])

        q = f"All {category} {property_}. {member.capitalize()} is a {category.rstrip('s')}. Does {member} {property_.replace('needs', 'need').replace('has', 'have').replace('takes', 'take')}?"
        thinking = (
            f"<think>\n"
            f"Let me reason through this:\n"
            f"Premise 1: All {category} {property_}.\n"
            f"Premise 2: {member.capitalize()} is a {category.rstrip('s')}.\n"
            f"Since {member} belongs to the category of {category}, and all {category} {property_}, "
            f"then {member} must also {property_.replace('needs', 'need').replace('has', 'have').replace('takes', 'take')}.\n"
            f"</think>\n\n"
            f"Yes. Since all {category} {property_}, and {member} is a {category.rstrip('s')}, "
            f"then {member} {property_.replace('needs', 'need').replace('has', 'have').replace('takes', 'take')}."
        )
        examples.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": thinking},
            ]
        })

    # --- Ordering / Sequencing ---
    for _ in range(count // 3):
        n = random.choice(names[:4])
        a = random.choice(animals)
        count_items = random.randint(2, 8)
        n2 = random.choice([x for x in names if x != n])
        count_items2 = random.randint(2, 8)

        if count_items > count_items2:
            who = n
        elif count_items2 > count_items:
            who = n2
        else:
            who = "Neither, they have the same number"

        q = f"{n} has {count_items} {a}. {n2} has {count_items2} {a}. Who has more {a}?"
        thinking = (
            f"<think>\n"
            f"Let me compare:\n"
            f"- {n} has {count_items} {a}\n"
            f"- {n2} has {count_items2} {a}\n"
            f"- {count_items} vs {count_items2}: "
        )
        if count_items > count_items2:
            thinking += f"{count_items} > {count_items2}, so {n} has more.\n</think>\n\n{n} has more {a} ({count_items} vs {count_items2})."
        elif count_items2 > count_items:
            thinking += f"{count_items2} > {count_items}, so {n2} has more.\n</think>\n\n{n2} has more {a} ({count_items2} vs {count_items})."
        else:
            thinking += f"they are equal.\n</think>\n\nThey have the same number of {a} ({count_items} each)."

        examples.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": thinking},
            ]
        })

    # --- If-then reasoning ---
    for _ in range(count // 3):
        condition = random.choice([
            ("it is raining", "the ground is wet", "Is the ground wet?"),
            ("the temperature is below 0°C", "water freezes", "Will water freeze?"),
            ("a number is even", "it is divisible by 2", "Is 14 divisible by 2?"),
            ("an animal is a mammal", "it is warm-blooded", "Is a dog warm-blooded?"),
            ("the sun has set", "it is dark outside", "Is it dark when the sun sets?"),
        ])
        premise, consequence, question = condition
        thinking = (
            f"<think>\n"
            f"Given rule: If {premise}, then {consequence}.\n"
            f"The question asks about a case where the condition is met.\n"
            f"Since the condition is satisfied, the consequence follows.\n"
            f"</think>\n\n"
            f"Yes. If {premise}, then {consequence}. Since the condition is met, "
            f"the answer is yes."
        )
        examples.append({
            "messages": [
                {"role": "user", "content": f"If {premise}, then {consequence}. {question}"},
                {"role": "assistant", "content": thinking},
            ]
        })

    random.shuffle(examples)
    return examples


def prepare_chat_data(output_path="data/colab_chat_train.jsonl", max_examples=50000):
    """
    Download instruction data for chat fine-tuning.
    Combines:
      - OpenOrca (general instructions)
      - Alpaca-GPT4 (clean instructions)
      - Synthetic chain-of-thought (math + logic reasoning)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        os.system(f"{sys.executable} -m pip install -q datasets")
        from datasets import load_dataset

    print(f"📥 Preparing instruction data (target: {max_examples} examples)...")
    os.makedirs(os.path.dirname(output_path) or "data", exist_ok=True)

    formatted = []

    # --- Source 1: Synthetic Chain-of-Thought (math + logic) ---
    # These go FIRST so they're always included, regardless of download failures
    print("  🧠 Generating chain-of-thought reasoning data...")
    cot_math = _generate_arithmetic_cot(count=5000)
    cot_logic = _generate_logic_cot(count=3000)
    formatted.extend(cot_math)
    formatted.extend(cot_logic)
    print(f"  ✅ CoT Math: {len(cot_math)} examples")
    print(f"  ✅ CoT Logic: {len(cot_logic)} examples")

    # --- Source 2: MetaMathQA (real math reasoning with step-by-step) ---
    print("  📂 Loading MetaMathQA...")
    try:
        ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
        math_count = 0
        for example in ds:
            query = example.get("query", "").strip()
            response = example.get("response", "").strip()
            if not query or not response:
                continue
            if len(response.split()) < 10:
                continue

            # Wrap response in think tags if it contains step-by-step work
            if "step" in response.lower() or "=" in response:
                answer_text = f"<think>\n{response}\n</think>"
            else:
                answer_text = response

            formatted.append({
                "messages": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": answer_text},
                ]
            })
            math_count += 1
            if math_count >= 5000:
                break

        print(f"  ✅ MetaMathQA: {math_count} examples")
    except Exception as e:
        print(f"  ⚠️ MetaMathQA failed: {e}")

    # --- Source 3: OpenOrca (general instructions) ---
    remaining_general = max_examples - len(formatted)
    orca_target = int(remaining_general * 0.65)

    print("  📂 Loading OpenOrca...")
    try:
        ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
        orca_count = 0
        for example in ds:
            system = example.get("system_prompt", "").strip()
            question = example.get("question", "").strip()
            answer = example.get("response", "").strip()

            if not question or not answer:
                continue
            if len(answer.split()) < 5:
                continue

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
            formatted.append({"messages": messages})
            orca_count += 1

            if orca_count >= orca_target:
                break

        print(f"  ✅ OpenOrca: {orca_count} examples")
    except Exception as e:
        print(f"  ⚠️ OpenOrca failed: {e}")

    # --- Source 4: Alpaca-GPT4 (clean instruction data) ---
    remaining = max_examples - len(formatted)
    print("  📂 Loading Alpaca-GPT4...")
    try:
        ds = load_dataset("vicgalle/alpaca-gpt4", split="train", streaming=True)
        alpaca_count = 0
        for example in ds:
            instruction = example.get("instruction", "").strip()
            inp = example.get("input", "").strip()
            output = example.get("output", "").strip()

            if not instruction or not output:
                continue
            if len(output.split()) < 5:
                continue

            user_msg = f"{instruction}\n{inp}".strip() if inp else instruction
            formatted.append({
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": output},
                ]
            })
            alpaca_count += 1

            if alpaca_count >= remaining:
                break

        print(f"  ✅ Alpaca-GPT4: {alpaca_count} examples")
    except Exception as e:
        print(f"  ⚠️ Alpaca-GPT4 failed: {e}")

    # --- Shuffle and write ---
    random.shuffle(formatted)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in formatted:
            f.write(json.dumps(record) + "\n")

    print(f"\n✅ Total: {len(formatted)} instruction examples saved to {output_path}")
    print(f"   📊 Data mix breakdown:")
    print(f"      - Chain-of-Thought (math+logic): ~{len(cot_math)+len(cot_logic)}")
    print(f"      - General instructions (OpenOrca+Alpaca): ~{len(formatted)-len(cot_math)-len(cot_logic)}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare datasets for Colab training")
    parser.add_argument("--pretrain", action="store_true", help="Download pretraining data")
    parser.add_argument("--chat", action="store_true", help="Download chat fine-tuning data")
    parser.add_argument("--pretrain-tokens-m", type=int, default=500,
                        help="Target pretraining data size in millions of tokens")
    parser.add_argument("--chat-examples", type=int, default=50000,
                        help="Target number of chat examples")
    parser.add_argument("--all", action="store_true", help="Download everything")
    args = parser.parse_args()

    if args.all or args.pretrain:
        prepare_pretraining_data(target_tokens_m=args.pretrain_tokens_m)
    if args.all or args.chat:
        prepare_chat_data(max_examples=args.chat_examples)
    if not (args.all or args.pretrain or args.chat):
        print("Usage: python prepare_colab_data.py --all")
        print("       python prepare_colab_data.py --pretrain --pretrain-tokens-m 500")
        print("       python prepare_colab_data.py --chat --chat-examples 50000")
