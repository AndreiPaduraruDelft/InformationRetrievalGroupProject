import re


def clean_query(q: str) -> str:
    return re.sub(r'[^\w\s]', ' ', q).strip()


def flanqr_reformulate(query: str, reformulator, instruction: str) -> str:
    keywords = reformulator.generate_keywords(instruction, query)
    return clean_query(query + " " + keywords)


def genqr_ensemble_reformulate(query: str, reformulator, instructions: list) -> str:
    parts = [query]
    for instruction in instructions:
        keywords = reformulator.generate_keywords(instruction, query)
        parts.append(keywords if keywords.strip() else query)
    return clean_query(" ".join(parts))
