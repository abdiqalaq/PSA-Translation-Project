import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

PLURAL_INSTITUTION_WORDS = {
    "authorities",
    "services",
    "teams",
    "programmes",
    "clinics",
    "hospitals",
    "departments",
    "centres",
    "facilities",
}


def capitalize_first(text: str) -> str:
    """Capitalize only the first letter without changing the rest."""
    text = str(text).strip()

    if not text:
        return text

    return text[0].upper() + text[1:]


def institution_is_plural(institution: str) -> bool:
    """Estimate whether an institution name requires a plural verb."""
    last_word = institution.lower().strip().split()[-1]

    return last_word in PLURAL_INSTITUTION_WORDS


def institution_verb(
    institution: str,
    singular: str,
    plural: str,
) -> str:
    """Return the correct verb based on the institution name."""
    if institution_is_plural(institution):
        return plural

    return singular


def action_for_group(action: str) -> str:
    """
    Convert second-person wording to wording suitable for a named group.

    Example:
    'cover your mouth' becomes 'cover their mouths'.
    """
    replacements = [
        (r"\byourself\b", "themselves"),
        (r"\byourselves\b", "themselves"),
        (r"\byour\b", "their"),
        (r"\byou\b", "they"),
    ]

    result = action

    for pattern, replacement in replacements:
        result = re.sub(
            pattern,
            replacement,
            result,
            flags=re.IGNORECASE,
        )

    specific_replacements = {
        "cover their mouth and nose": "cover their mouths and noses",
        "wash their hand": "wash their hands",
        "keep their medical appointment date": (
            "keep their medical appointment dates"
        ),
    }

    for old, new in specific_replacements.items():
        result = result.replace(old, new)

    return result


def clean_generated_text(text: str) -> str:
    """Apply basic grammar, spacing, and capitalization corrections."""

    text = re.sub(r"\s+", " ", text).strip()

    # Remove spaces before punctuation.
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    # Ensure one space follows punctuation.
    text = re.sub(r"([.!?])(?=[A-Za-z])", r"\1 ", text)

    # Capitalize the beginning and the first letter after sentence endings.
    text = capitalize_first(text)

    text = re.sub(
        r"([.!?]\s+)([a-z])",
        lambda match: (
            match.group(1) + match.group(2).upper()
        ),
        text,
    )

    # Fix common generated grammar problems.
    replacements = {
        "Ministry of Health advise ": "Ministry of Health advises ",
        "Ministry of Health remind ": "Ministry of Health reminds ",
        "Ministry of Health urge ": "Ministry of Health urges ",
        "World Health Organization advise ": (
            "World Health Organization advises "
        ),
        "World Health Organization remind ": (
            "World Health Organization reminds "
        ),
        "Kenya Red Cross Society advise ": (
            "Kenya Red Cross Society advises "
        ),
        "public hospitals advises ": "public hospitals advise ",
        "public hospitals reminds ": "public hospitals remind ",
        "public hospitals urges ": "public hospitals urge ",
        "mobile health clinics advises ": (
            "mobile health clinics advise "
        ),
        "mobile health clinics reminds ": (
            "mobile health clinics remind "
        ),
        "child welfare clinics advises ": (
            "child welfare clinics advise "
        ),
        "nutrition support programmes advises ": (
            "nutrition support programmes advise "
        ),
        "nutrition support programmes reminds ": (
            "nutrition support programmes remind "
        ),
        "local health authorities advises ": (
            "local health authorities advise "
        ),
        "community health services advises ": (
            "community health services advise "
        ),
        "disease surveillance teams advises ": (
            "disease surveillance teams advise "
        ),
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Capitalize known institutions when they begin a sentence.
    institution_names = [
        "public hospitals",
        "mobile health clinics",
        "child welfare clinics",
        "nutrition support programmes",
        "local health authorities",
        "community health services",
        "disease surveillance teams",
        "licensed counselling centres",
        "maternal health clinics",
    ]

    for institution in institution_names:
        text = re.sub(
            rf"(^|[.!?]\s+){re.escape(institution)}",
            lambda match: (
                match.group(1) + capitalize_first(institution)
            ),
            text,
        )

    if text and text[-1] not in ".!?":
        text += "."

    return text

DEFAULT_DOMAIN = "health"
DEFAULT_NUMBER_OF_PSAS = 10_000
DEFAULT_SEED = 42

KNOWLEDGE_BASE_DIR = Path("knowledge_base")
TEMPLATES_DIR = Path("templates")
OUTPUT_DIR = Path("outputs")

REQUIRED_FIELDS = {
    "institutions",
    "audiences",
    "hazards",
    "actions",
    "locations",
    "terminology",
}


def load_json(file_path: Path) -> dict:
    """Load and return a JSON file."""

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {file_path}: "
            f"line {error.lineno}, column {error.colno}"
        ) from error


def normalize_text(text: str) -> str:
    """Normalize text for duplicate detection."""

    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    return text

def clean_psa_text(text: str) -> str:
    """Clean spacing and punctuation in a generated PSA."""

    text = re.sub(r"\s+", " ", text).strip()

    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    # Ensure a space after punctuation
    text = re.sub(r"([.,!?;:])(?=[A-Za-z])", r"\1 ", text)

    # Collapse repeated punctuation
    text = re.sub(r"\.{2,}", ".", text)

    # Fix known joined-word issue
    text = text.replace("byviolence", "by violence")

    if text and text[-1] not in ".!?":
        text += "."

    return text



def capitalize_first(text: str) -> str:
    """Capitalize the first character without changing the rest."""

    text = text.strip()

    if not text:
        return text

    return text[0].upper() + text[1:]


def validate_knowledge_base(knowledge_base: dict) -> None:
    """Check that the knowledge base has the required structure."""

    if "domain" not in knowledge_base:
        raise ValueError("Knowledge base is missing the 'domain' field.")

    if "subcategories" not in knowledge_base:
        raise ValueError(
            "Knowledge base is missing the 'subcategories' field."
        )

    subcategories = knowledge_base["subcategories"]

    if not isinstance(subcategories, dict) or not subcategories:
        raise ValueError("The knowledge base contains no subcategories.")

    for subcategory, content in subcategories.items():
        missing_fields = REQUIRED_FIELDS - set(content.keys())

        if missing_fields:
            raise ValueError(
                f"Subcategory '{subcategory}' is missing fields: "
                f"{sorted(missing_fields)}"
            )

        for field in REQUIRED_FIELDS:
            values = content[field]

            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"'{field}' in '{subcategory}' must be a non-empty list."
                )

            if not all(
                isinstance(value, str) and value.strip()
                for value in values
            ):
                raise ValueError(
                    f"'{field}' in '{subcategory}' contains invalid values."
                )


def validate_templates(template_data: dict, subcategories: dict) -> None:
    """Validate that every subcategory has its own templates."""

    templates = template_data.get("templates")

    if not isinstance(templates, dict):
        raise ValueError(
            "'templates' must be a dictionary keyed by subcategory."
        )

    for subcategory in subcategories:

        if subcategory not in templates:
            raise ValueError(
                f"Missing templates for '{subcategory}'."
            )

        if (
            not isinstance(templates[subcategory], list)
            or not templates[subcategory]
        ):
            raise ValueError(
                f"'{subcategory}' must contain a non-empty template list."
            )

        if not all(
            isinstance(t, str) and t.strip()
            for t in templates[subcategory]
        ):
            raise ValueError(
                f"Invalid templates found in '{subcategory}'."
            )
def choose_components(
    knowledge: dict,
    rng: random.Random,
) -> dict:
    """Select PSA components and prepare grammar-aware versions."""

    rules = knowledge.get("rules", [])

    if rules:
        selected = rng.choice(rules)

        institution = rng.choice(selected["institutions"])
        audience = rng.choice(selected["audiences"])
        action = selected["action"]
        hazard = rng.choice(selected["hazards"])
        location = rng.choice(selected["locations"])
        terminology = rng.choice(selected["terminology"])

    else:
        institution = rng.choice(knowledge["institutions"])
        audience = rng.choice(knowledge["audiences"])
        action = rng.choice(knowledge["actions"])
        hazard = rng.choice(knowledge["hazards"])
        location = rng.choice(knowledge["locations"])
        terminology = rng.choice(knowledge["terminology"])

    return {
        "institution": institution,
        "institution_cap": capitalize_first(institution),

        "institution_advises": institution_verb(
            institution,
            singular="advises",
            plural="advise",
        ),

        "institution_reminds": institution_verb(
            institution,
            singular="reminds",
            plural="remind",
        ),

        "institution_urges": institution_verb(
            institution,
            singular="urges",
            plural="urge",
        ),

        "institution_warns": institution_verb(
    institution,
    singular="warns",
    plural="warn",
),

"institution_encourages": institution_verb(
    institution,
    singular="encourages",
    plural="encourage",
),

"institution_recommends": institution_verb(
    institution,
    singular="recommends",
    plural="recommend",
),

        "audience": audience,
        "audience_cap": capitalize_first(audience),

        "hazard": hazard,
        "hazard_cap": capitalize_first(hazard),

        "action": action,
        "action_cap": capitalize_first(action),

        # Use this when a named audience comes before the action.
        "action_for_audience": action_for_group(action),

        "location": location,
        "location_cap": capitalize_first(location),

        "terminology": terminology,
        "terminology_cap": capitalize_first(terminology),
    }

def generate_one_psa(
    knowledge: dict,
    templates: list[str],
    rng: random.Random,
) -> str:
    """Generate one PSA from a template and knowledge-base values."""

    components = choose_components(knowledge, rng)
    template = rng.choice(templates)

    try:
        psa_text = template.format(**components)
        psa_text = clean_generated_text(psa_text)

    except KeyError as error:
        raise ValueError(
            f"Template contains an unsupported placeholder: {error}"
        ) from error

    return clean_psa_text(psa_text)


def generate_psas(
    knowledge_base: dict,
    template_data: dict,
    number_of_psas: int,
    seed: int,
) -> pd.DataFrame:
    """Generate a unique PSA dataset."""

    if number_of_psas <= 0:
        raise ValueError("The number of PSAs must be greater than zero.")

    validate_knowledge_base(knowledge_base)
    validate_templates(
    template_data,
    knowledge_base["subcategories"],
)

    domain = knowledge_base["domain"]
    subcategories = knowledge_base["subcategories"]
    templates_by_subcategory = template_data["templates"]

    rng = random.Random(seed)

    records = []
    used_sentences = set()

    subcategory_names = list(subcategories.keys())

    maximum_attempts = number_of_psas * 200
    attempts = 0
    duplicate_attempts = 0

    while len(records) < number_of_psas:
        attempts += 1

        if attempts > maximum_attempts:
            raise RuntimeError(
                f"Unable to generate {number_of_psas:,} unique PSAs. "
                "Expand the knowledge base or add more templates."
            )

        subcategory = rng.choice(subcategory_names)

        knowledge = subcategories[subcategory]

        templates = templates_by_subcategory[subcategory]

        psa_text = generate_one_psa(
            knowledge=knowledge,
            templates=templates,
            rng=rng,
      )
        
        normalized_text = normalize_text(psa_text)

        if normalized_text in used_sentences:
            duplicate_attempts += 1
            continue

        used_sentences.add(normalized_text)

        records.append(
            {
                "psa_id": f"PSA_{len(records) + 1:05d}",
                "domain": domain,
                "subcategory": subcategory,
                "psa_text": psa_text,
            }
        )

        if len(records) % 1_000 == 0:
            print(f"Generated {len(records):,}/{number_of_psas:,} PSAs")

    print(f"Duplicate attempts skipped: {duplicate_attempts:,}")

    return pd.DataFrame(records)


def save_dataset(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save the generated dataset to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )


def display_summary(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Print dataset statistics and sample PSAs."""

    print("\nGeneration completed successfully.")
    print(f"Total PSAs: {len(dataframe):,}")
    print(f"Unique PSAs: {dataframe['psa_text'].nunique():,}")
    print(f"Output file: {output_path.resolve()}")

    print("\nPSAs by subcategory:")
    print(
        dataframe["subcategory"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nSample PSAs:\n")

    for _, row in dataframe.head(10).iterrows():
        print(f"{row['psa_id']} | {row['subcategory']}")
        print(row["psa_text"])
        print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Public Service Announcements."
    )

    parser.add_argument(
        "--domain",
        type=str,
        default=DEFAULT_DOMAIN,
        help="Domain name matching the JSON filenames.",
    )

    parser.add_argument(
        "--number",
        type=int,
        default=DEFAULT_NUMBER_OF_PSAS,
        help="Number of unique PSAs to generate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible generation.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional custom output CSV path.",
    )

    args = parser.parse_args()

    domain_key = args.domain.lower().strip()

    knowledge_base_file = (
        KNOWLEDGE_BASE_DIR / f"{domain_key}.json"
    )

    templates_file = (
        TEMPLATES_DIR / f"{domain_key}_templates.json"
    )

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            OUTPUT_DIR / f"{domain_key}_psas_{args.number}.csv"
        )

    knowledge_base = load_json(knowledge_base_file)
    template_data = load_json(templates_file)

    dataframe = generate_psas(
        knowledge_base=knowledge_base,
        template_data=template_data,
        number_of_psas=args.number,
        seed=args.seed,
    )

    save_dataset(
        dataframe=dataframe,
        output_path=output_path,
    )

    display_summary(
        dataframe=dataframe,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()