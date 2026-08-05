from config import Settings


def test_ensure_directories_creates_runtime_directories(tmp_path):
    """Settings should create all runtime folders when they are absent."""
    settings = Settings(
        data_dir=tmp_path / "data",
        vectorstore_dir=tmp_path / "vectorstore",
        log_dir=tmp_path / "logs",
    )

    settings.ensure_directories()

    assert settings.data_dir.is_dir()
    assert settings.vectorstore_dir.is_dir()
    assert settings.log_dir.is_dir()


def test_settings_uses_expected_defaults():
    """Default values should keep the project behavior predictable."""
    settings = Settings()

    assert settings.default_currency == "USD"
    assert settings.log_level == "INFO"
    assert settings.data_dir.name == "data"
    assert settings.vectorstore_dir.name == "vectorstore"