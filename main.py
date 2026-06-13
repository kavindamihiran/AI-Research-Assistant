from dotenv import load_dotenv

from core import (
    PROVIDER_ENV_KEYS,
    PROVIDER_NVIDIA_NIM,
    PROVIDER_OPENROUTER,
    get_api_key,
    get_default_model,
    get_provider_options,
    perform_research,
    provider_help_url,
)


load_dotenv(override=True)


PROVIDERS = [PROVIDER_NVIDIA_NIM, PROVIDER_OPENROUTER]


def choose_from_list(title: str, values: list[str], default_index: int = 0) -> str:
    print(f"\n{title}")
    print("-" * 72)
    for index, value in enumerate(values, 1):
        print(f"  [{index}] {value}")
    print("-" * 72)

    while True:
        choice = input(f"Select 1-{len(values)} or press Enter for [{default_index + 1}]: ").strip()
        if not choice:
            return values[default_index]
        if choice.isdigit() and 1 <= int(choice) <= len(values):
            return values[int(choice) - 1]
        print("Invalid choice. Try again.")


def select_provider() -> str:
    return choose_from_list("Providers", PROVIDERS)


def select_model(provider: str) -> str:
    options = get_provider_options(provider)
    labels = [f"{option.label} ({option.model_id})" for option in options]
    selected = choose_from_list("Models", labels)
    selected_index = labels.index(selected)
    model_id = options[selected_index].model_id

    custom = input("Custom model ID (optional, press Enter to use selected): ").strip()
    return custom or model_id or get_default_model(provider)


def main() -> None:
    print("=" * 72)
    print("AI Research Assistant - Command Line")
    print("=" * 72)

    provider = select_provider()
    api_key = get_api_key(provider)
    env_key = PROVIDER_ENV_KEYS[provider]

    if not api_key:
        print(f"\nMissing {env_key}.")
        print(f"Create a key at: {provider_help_url(provider)}")
        print(f"Add it to .env as: {env_key}=your-key-here")
        raise SystemExit(1)

    model_name = select_model(provider)
    query = input("\nWhat can I help you research? ").strip()

    if not query:
        print("No query provided. Exiting.")
        raise SystemExit(1)

    print("\nResearching...\n")

    try:
        result = perform_research(provider, api_key, model_name, query)
    except Exception as exc:
        message = str(exc)
        print(f"Research failed: {message}")
        if "429" in message:
            print("Tip: this usually means the provider or model rate limit was reached.")
        elif "tool" in message.lower():
            print("Tip: try a model that supports tool calling, or use a custom model ID from the provider catalog.")
        raise SystemExit(1)

    print("=" * 72)
    print("RESEARCH COMPLETE")
    print("=" * 72)
    print(f"\nTopic: {result.topic}")
    print(f"Confidence: {result.confidence}")
    print(f"\nSummary:\n{result.summary}")

    if result.key_findings:
        print("\nKey Findings:")
        for item in result.key_findings:
            print(f"  - {item}")

    if result.sources:
        print("\nSources:")
        for index, source in enumerate(result.sources, 1):
            print(f"  {index}. {source}")

    if result.suggested_followups:
        print("\nSuggested Follow-ups:")
        for item in result.suggested_followups:
            print(f"  - {item}")

    if result.tools_used:
        print(f"\nTools Used: {', '.join(result.tools_used)}")


if __name__ == "__main__":
    main()
