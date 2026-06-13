"""
API key encryption at rest using Fernet (symmetric AES-128-CBC).
Falls back to base64 obfuscation if cryptography is not installed.

Keys are stored encrypted in the SQLite database and only decrypted
in-memory when needed for API calls.
"""
import os
import base64
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from cryptography.fernet import Fernet
    HAS_FERNET = True
except ImportError:
    Fernet = None
    HAS_FERNET = False


class KeyEncryption:
    """
    Encrypts/decrypts API keys using Fernet (cryptography package)
    with a key derived from a machine-local secret file.
    """

    KEY_FILE = os.path.expanduser("~/.colab-agent/encryption.key")
    KEY_ENV_VAR = "COLAB_AGENT_ENCRYPTION_KEY"

    def __init__(self, custom_key: Optional[str] = None):
        self._fernet: Optional[Fernet] = None
        if HAS_FERNET:
            key = custom_key or os.environ.get(self.KEY_ENV_VAR)
            if key:
                self._fernet = Fernet(key.encode() if isinstance(key, str) else key)
            elif os.path.exists(self.KEY_FILE):
                try:
                    with open(self.KEY_FILE, "rb") as f:
                        self._fernet = Fernet(f.read().strip())
                except Exception as e:
                    logger.warning(f"Failed to load encryption key from {self.KEY_FILE}: {e}")

    def _ensure_key(self):
        if self._fernet is not None:
            return
        if HAS_FERNET:
            key = Fernet.generate_key()
            os.makedirs(os.path.dirname(self.KEY_FILE), exist_ok=True)
            try:
                with open(self.KEY_FILE, "wb") as f:
                    f.write(key)
                logger.info(f"Generated new encryption key at {self.KEY_FILE}")
            except Exception as e:
                logger.warning(f"Cannot write key file {self.KEY_FILE}: {e}")
            self._fernet = Fernet(key)

    def encrypt(self, plaintext: str) -> str:
        if not plaintext:
            return ""
        if HAS_FERNET:
            self._ensure_key()
            return self._fernet.encrypt(plaintext.encode()).decode()
        return _obfuscate(plaintext)

    def decrypt(self, ciphertext: str) -> Optional[str]:
        if not ciphertext:
            return None
        if HAS_FERNET:
            try:
                if self._fernet is None:
                    self._ensure_key()
                return self._fernet.decrypt(ciphertext.encode()).decode()
            except Exception as e:
                logger.warning(f"Decryption failed: {e}")
                return None
        return _deobfuscate(ciphertext)

    @staticmethod
    def has_crypto() -> bool:
        return HAS_FERNET


def _obfuscate(plain: str) -> str:
    """Fallback obfuscation when cryptography is not available."""
    encoded = base64.b64encode(plain.encode()).decode()
    return f"obf:{encoded[::-1]}"


def _deobfuscate(cipher: str) -> Optional[str]:
    if not cipher:
        return None
    if cipher.startswith("obf:") and HAS_FERNET is not None:
        return None
    if cipher.startswith("obf:"):
        try:
            return base64.b64decode(cipher[4:][::-1]).decode()
        except Exception:
            return None
    return cipher
