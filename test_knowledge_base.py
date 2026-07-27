import json
from pathlib import Path


HEALTH_FILE = Path("knowledge_base/health.json")


def load_health_knowledge_base() -> dict:
    if not HEALTH_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge-base file not found: {HEALTH_FILE}"
        )

    with HEALTH_FILE.open("r", encoding="utf-8") as file:
        knowledge_base = json.load(file)

    return knowledge_base


def main() -> None:
    knowledge_base = load_health_knowledge_base()

    domain = knowledge_base["domain"]
    subcategories = knowledge_base["subcategories"]

    print(f"Domain: {domain}")
    print(f"Number of subcategories: {len(subcategories)}")

    for name, content in subcategories.items():
        print(f"\n{name}")
        print(f"  Institutions: {len(content['institutions'])}")
        print(f"  Audiences: {len(content['audiences'])}")
        print(f"  Hazards: {len(content['hazards'])}")
        print(f"  Actions: {len(content['actions'])}")
        print(f"  Locations: {len(content['locations'])}")
        print(f"  Terminology: {len(content['terminology'])}")


if __name__ == "__main__":
    main()