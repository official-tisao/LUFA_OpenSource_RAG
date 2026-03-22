    "en": (
        "You are a helpful legal assistant specializing in university collective agreements. "
        "Answer ONLY using the provided context. Cite the source document name and page for every claim. "
        "If the context does not contain the answer, say so clearly. Respond in English."
    ),
    "fr": (
        "Tu es un assistant juridique spécialisé dans les conventions collectives universitaires. "
        "Réponds UNIQUEMENT à partir du contexte fourni. Cite le document source et la page pour chaque affirmation. "
        "Si le contexte ne contient pas la réponse, dis-le clairement. Réponds en français."
    ),
}


class CopilotEngine:
    """
    Wraps the GitHub Models API (OpenAI-compatible) to use frontier models
    for the generation step after local ChromaDB retrieval.
    """

    def __init__(
        self,
        model:        str = "gpt-4o",
        github_token: Optional[str] = None,
        config_path:  str = "config/config.yaml",
    ):
        self.model = MODEL_ALIASES.get(model, model)
        token = (
            github_token
            or os.environ.get("GITHUB_TOKEN")
            or self._load_token_from_config(config_path)
        )
        if not token:
            raise EnvironmentError(
                "GITHUB_TOKEN not found. Set it as an environment variable or in config.yaml "
                "under copilot.github_token. Get a token at https://github.com/settings/tokens"
            )

        self.client = OpenAI(
            base_url=GITHUB_MODELS_ENDPOINT,
            api_key=token,
        )
        print(f"[CopilotEngine] Initialized with model: {self.model}")

    @staticmethod
    def _load_token_from_config(path: str) -> Optional[str]:
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("copilot", {}).get("github_token")
        except FileNotFoundError:
